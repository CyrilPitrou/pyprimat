"""
Tests for the GUI's "Manage networks" dialog and the "Create custom network"
dialog it gates (``primat.gui.params_form``).

These drive the actual popup workflow end to end with Streamlit's ``AppTest``
harness (no browser needed), mimicking the user flows a person would actually
click through: opening "Manage networks", listing/removing/renaming networks,
loading one from a zip, handing off to "Create new network" to build one from
scratch, toggling reactions on/off (individually and via Select all/Deselect
all), switching the base network, editing a decay rate, adding a brand-new
reaction, and applying the result. Each is a regression guard for a bug found
and fixed by hand-testing the popup:

* the dialogs not being mutually exclusive after one was dismissed via
  its own close button;
* Select all/Deselect all having no visible effect, and switching the base
  network not refreshing the category list, both caused by Streamlit widgets
  silently keeping their own stale state once a key has been used once
  (``_DialogState.bump_gen``);
* reactions above the dialog's own ``amax`` silently staying in the solved
  network because ``removed`` was computed against the *filtered* view
  instead of the full large-network list (``_DialogState.to_custom_network``);
* a decay reaction showing a misleading "(no table)" instead of its own
  editable rate (``_render_decay_category``/``_render_decay_row``);
* a brand-new "Add new rate" reaction showing "(no table)" in its own
  category despite having an uploaded table (``_render_reaction_row``);
* "Create this network" silently re-running BBN instead of just saving and
  returning to "Manage networks" (the old "Apply and run BBN" behaviour);
* re-editing a previously built/imported custom network mislabelling every
  one of its unmodified, shipped-default tables as "_custom" (``reset``);
* the sidebar's "Limit max mass number" filter silently carrying an
  unrelated previous network's choice over onto a freshly chosen custom
  network (``amax_prev_network``); it stays clickable (amax does filter a
  custom network's own kept-list too) but starts unchecked again on every
  genuine transition into a (possibly different) custom network;
* loading a network under a reserved built-in name ("small"/"large"/
  "small_parthenope") silently shadowing the real network of that name.

A harness quirk worth knowing if extending these tests: once the "Create
custom network" dialog has been opened and then closed again (via its own
``st.rerun()``), this Streamlit version's ``AppTest`` raises a spurious
``KeyError`` on collecting widget state on the *next* ``.run()`` -- even with
no further click -- because it tries to re-collect a stale per-reaction
widget that no longer renders. So every test below either (a) never closes
the "Create custom network" dialog once opened (interacts with it freely
while it stays open, never triggers "Create this network"), or (b) closes it
once and asserts immediately with no further ``.run()`` call, or (c) avoids
opening it at all and seeds ``_known_custom_networks`` directly via
``at.session_state`` for scenarios that need a custom network already in
place (mirroring the equivalent workaround already used by the import-
round-trip test below).

All tests are skipped if the optional ``gui`` extra is not installed, and
``network="large"``-based ones additionally skip if the AC2024 data has not
been generated (mirrors ``test_gui.py``).
"""
import os

import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

pytestmark = [pytest.mark.slow, pytest.mark.solve, pytest.mark.gui]

APP_PATH = "primat/gui/app.py"

_AC2024_DIR = os.path.join(os.path.dirname(__file__), "..", "primat",
                           "rates", "nuclear", "data")
_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/nuclear/data not generated",
)


def _open_manage_dialog(at):
    [btn] = [b for b in at.sidebar.button if b.label == "Manage networks"]
    btn.click()
    at.run(timeout=60)
    return at


def _open_create_dialog(at):
    """Open "Manage networks", then hand off to "Create new network".

    The resulting dialog stays open for the rest of the test -- see the
    module docstring's harness-quirk note for why no test calls
    "Create this network" and then keeps interacting afterwards.
    """
    _open_manage_dialog(at)
    [btn] = [b for b in at.button if b.key == "_btn_create_new_network"]
    btn.click()
    at.run(timeout=60)
    return at


def _toggle_keep(at, bare_name, value):
    """Flip a reaction row's "keep" toggle, regardless of its current `_dialog_gen`
    suffix (the key embeds a generation counter that changes whenever the
    backing state is rebuilt -- see ``_DialogState.bump_gen``)."""
    [t] = [t for t in at.toggle if t.key and t.key.endswith(f"_{bare_name}")
          and t.key.startswith("_dialog_keep_")]
    t.set_value(value)
    at.run(timeout=60)
    return at


def _click_create_this_network(at):
    """Click the "Create custom network" dialog's footer button.

    Per the module docstring's harness-quirk note, no test calls this and
    then performs any further ``.run()`` -- assert on the resulting
    ``at.session_state`` immediately.
    """
    [btn] = [b for b in at.button if b.key == "_dialog_apply"]
    btn.click()
    at.run(timeout=60)
    return at


# ---------------------------------------------------------------------------
# Single entry point: "Manage networks"
# ---------------------------------------------------------------------------

def test_manage_button_opens_dialog_and_hands_off_to_create():
    """There is exactly one sidebar button now ("Manage networks"); clicking
    it opens the management dialog, and its "Create new network" button
    hands off to the "Create custom network" dialog, closing the former."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert [b.label for b in at.sidebar.button] == ["Manage networks"]

    _open_manage_dialog(at)
    assert not at.exception
    assert at.session_state["_show_manage_dialog"] is True
    assert at.session_state["_show_custom_dialog"] is False
    # The network list is a single radio (no separate, duplicate listing);
    # "small" is selected by default and, being built-in, offers no Remove.
    [radio] = [r for r in at.radio if r.key == "_manage_selected_network"]
    assert any(opt.startswith("small (") for opt in radio.options)
    assert at.session_state["_manage_selected_network"] == "small"
    assert not [b for b in at.button if b.key == "_manage_remove_small"]

    [create_btn] = [b for b in at.button if b.key == "_btn_create_new_network"]
    create_btn.click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["_show_custom_dialog"] is True
    assert at.session_state["_show_manage_dialog"] is False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_dialog_defaults_to_small_amax7():
    """Opening "Create custom network" for the first time this session
    proposes 'small' with 'Limit A' enabled at amax=7."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    assert not at.exception
    assert at.session_state["_dialog_base_network"] == "small"
    assert at.session_state["_dialog_amax_enabled"] is True
    assert at.session_state["_dialog_amax_value"] == 7
    # small has 12 reactions, all A <= 7, so amax=7 changes nothing here.
    assert sum(1 for v in at.session_state["_dialog_keep"].values() if v) == 12


# ---------------------------------------------------------------------------
# Select all / Deselect all
# ---------------------------------------------------------------------------

def test_select_all_and_deselect_all_actually_toggle_reactions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)

    # Category 2 (A<=2) contains n_p__d_g for the small/amax7 base.
    [desel_cat2] = [b for b in at.button if b.key == "_dialog_deselall_2"]
    desel_cat2.click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["_dialog_keep"].get("n_p__d_g") is False

    [sel_cat2] = [b for b in at.button if b.key == "_dialog_selall_2"]
    sel_cat2.click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["_dialog_keep"].get("n_p__d_g") is True


def test_select_all_keeps_its_category_expanded():
    """Clicking Select all/Deselect all must not collapse the category it's
    in (the user is mid-edit there)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)

    [desel_cat2] = [b for b in at.button if b.key == "_dialog_deselall_2"]
    desel_cat2.click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["_dialog_expander_cat_2"] is True


# ---------------------------------------------------------------------------
# Switching the base network hard-refreshes the category list
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_switching_base_network_refreshes_kept_set():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    assert sum(1 for v in at.session_state["_dialog_keep"].values() if v) == 12

    # Re-fetch the selectbox after every .run(): reusing a reference captured
    # before a previous rerun silently no-ops the next .set_value() (AppTest
    # binds it to that run's own snapshot of the element tree).
    [base_select] = [s for s in at.selectbox if s.key == "_dialog_base_network"]
    base_select.set_value("large")
    at.run(timeout=60)
    assert not at.exception
    # "large" with amax=7 still on has more than 12 reactions.
    n_kept = sum(1 for v in at.session_state["_dialog_keep"].values() if v)
    assert n_kept > 12

    [base_select] = [s for s in at.selectbox if s.key == "_dialog_base_network"]
    base_select.set_value("small")
    at.run(timeout=60)
    assert not at.exception
    assert sum(1 for v in at.session_state["_dialog_keep"].values() if v) == 12


# ---------------------------------------------------------------------------
# amax correctness: nothing above the dialog's amax leaks into the network
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_amax_filtered_reactions_are_actually_removed_on_create():
    """Regression guard: reactions above the dialog's own amax must end up in
    custom_network["removed"], not silently survive because they were never
    shown (and so never explicitly toggled off). "Create this network" no
    longer runs BBN -- inspect the saved ``_known_custom_networks`` entry
    directly instead of the (now nonexistent) post-solve ``params``."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)

    [base_select] = [s for s in at.selectbox if s.key == "_dialog_base_network"]
    base_select.set_value("large")
    at.run(timeout=60)
    [amax_value] = [n for n in at.number_input if n.key == "_dialog_amax_value"]
    amax_value.set_value(8)
    at.run(timeout=60)
    assert not at.exception

    _click_create_this_network(at)
    assert not at.exception

    from primat.network_data import reaction_category
    saved = at.session_state["_known_custom_networks"]["custom"]
    custom_network = saved["custom_network"]
    # This is the actual regression this guards: reactions the dialog never
    # even showed (everything above amax=8) must still land in "removed",
    # not be silently kept just because the user never saw a toggle for them.
    above_amax = [n for n in custom_network["removed"]
                 if n not in custom_network.get("added", {})
                 and reaction_category(n) > 8]
    assert above_amax, "no above-amax reaction found in 'removed' -- amax leak regressed"
    # large+amax8 == the old "medium" network's 68 reactions (67 + n__p);
    # describe_reactions() includes the prepended n__p.
    assert len(saved["kept"]) == 67
    # Saving must not have run BBN or activated the network in the sidebar.
    assert "params" not in at.session_state
    assert "_active_custom_network" not in at.session_state
    assert at.session_state["_show_manage_dialog"] is True
    assert at.session_state["_manage_selected_network"] == "custom"


# ---------------------------------------------------------------------------
# Removing a single reaction
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_removing_one_reaction_and_creating_drops_it_from_the_network():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    [base_select] = [s for s in at.selectbox if s.key == "_dialog_base_network"]
    base_select.set_value("large")
    at.run(timeout=60)
    [amax_value] = [n for n in at.number_input if n.key == "_dialog_amax_value"]
    amax_value.set_value(8)
    at.run(timeout=60)

    _toggle_keep(at, "d_d__t_p", False)
    assert not at.exception
    assert at.session_state["_dialog_keep"]["d_d__t_p"] is False

    _click_create_this_network(at)
    assert not at.exception

    saved = at.session_state["_known_custom_networks"]["custom"]
    assert "d_d__t_p" in saved["custom_network"]["removed"]
    # 67 reactions at amax=8 minus 1 removed = 66 (the dialog's own "kept"
    # list never includes the prepended n__p, unlike the solved subheader).
    assert len(saved["kept"]) == 66


# ---------------------------------------------------------------------------
# Decays get their own category with an editable rate, not "(no table)"
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_decay_reactions_have_their_own_editable_category():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    [base_select] = [s for s in at.selectbox if s.key == "_dialog_base_network"]
    base_select.set_value("large")
    at.run(timeout=60)

    decay_expanders = [e for e in at.expander if e.label.startswith("Decays")]
    assert decay_expanders, "no 'Decays' category rendered"

    decay_inputs = [n for n in at.number_input if n.key.startswith("_dialog_decay_rate_")]
    assert decay_inputs, "no editable decay-rate inputs rendered"

    # Editing one away from its shipped value must register as an override.
    one = decay_inputs[0]
    new_value = one.value * 2.0
    one.set_value(new_value)
    at.run(timeout=60)
    assert not at.exception
    # Key form: _dialog_decay_rate_{gen}_{name}; name may itself contain
    # underscores, so split off the two known prefix parts instead.
    prefix = "_dialog_decay_rate_"
    rest = one.key[len(prefix):]
    _gen, name = rest.split("_", 1)
    assert at.session_state["_dialog_decay_override"].get(name) == pytest.approx(new_value)


# ---------------------------------------------------------------------------
# Add a brand-new reaction: it must show up with its table, not "(no table)"
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_added_reaction_table_is_recognised_in_its_category():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)

    [open_btn] = [b for b in at.button if b.key == "_dialog_add_rate_open_btn"]
    open_btn.click()
    at.run(timeout=60)
    assert not at.exception

    # A fictional but balanced reaction not already in the shipped catalog
    # (He3_t__a_d etc. are real BBN channels already in large.txt).
    [name_in] = [t for t in at.text_input if t.key == "_dialog_add_name"]
    name_in.set_value("Li6_d__a_a")
    at.run(timeout=60)
    assert not at.exception

    [uploader] = [u for u in at.file_uploader if u.key == "_dialog_add_table"]
    table_text = (
        "# ref=test\n"
        + "\n".join(f"{t9:.6e} 1.0 0.1" for t9 in (1e-3, 1e-2, 1e-1, 1.0, 10.0))
    )
    uploader.set_value(("rate.txt", table_text.encode(), "text/plain"))
    at.run(timeout=60)

    [add_btn] = [b for b in at.button if b.key == "_dialog_add_submit"]
    assert not add_btn.disabled
    add_btn.click()
    at.run(timeout=60)
    assert not at.exception

    # The popup form must have collapsed (dismissed) after a successful add.
    assert at.session_state["_dialog_add_rate_open"] is False
    assert "Li6_d__a_a" in at.session_state["_dialog_added"]
    assert at.session_state["_dialog_keep"]["Li6_d__a_a"] is True
    # The chosen basename must read as a custom override, not the bare
    # upload name (CPLAN.md request).
    assert at.session_state["_dialog_table_choice"]["Li6_d__a_a"] == "Li6_d__a_a_custom_rate.txt"

    # And, the actual regression this guards: its category row must show the
    # uploaded table, not "(no table)".
    captions = [c.value for c in at.caption if c.value == "(no table)"]
    assert not captions


# ---------------------------------------------------------------------------
# "Show evolved nuclides" summary
# ---------------------------------------------------------------------------

def test_evolved_nuclides_section_matches_small_network():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    [evolved] = [e for e in at.expander if e.label.startswith("Show evolved nuclides")]
    assert evolved.label == "Show evolved nuclides (8)"


# ---------------------------------------------------------------------------
# "Create this network" saves and returns to "Manage networks" without
# running BBN (FUTURE.md / CPLAN.md request)
# ---------------------------------------------------------------------------

def test_create_this_network_saves_without_running_bbn():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    [apply_btn] = [b for b in at.button if b.key == "_dialog_apply"]
    assert apply_btn.label == "Create this network"
    _toggle_keep(at, "d_d__t_p", False)
    _click_create_this_network(at)
    assert not at.exception

    # Saved under the default title ("custom"), but not yet run or selected.
    assert "custom" in at.session_state["_known_custom_networks"]
    assert "params" not in at.session_state
    assert "_active_custom_network" not in at.session_state
    # Back on "Manage networks", with the just-created network preselected.
    assert at.session_state["_show_manage_dialog"] is True
    assert at.session_state["_show_custom_dialog"] is False


# ---------------------------------------------------------------------------
# Seeded-network flows: Manage -> Close -> Run BBN (avoids the AppTest
# dialog-reopen quirk documented in the module docstring)
# ---------------------------------------------------------------------------

def _seed_known_network(at, title, kept, custom_network):
    at.session_state["_known_custom_networks"] = {
        title: {"kept": list(kept), "tables": {}, "custom_network": custom_network},
    }


def test_close_stages_selection_for_the_sidebar():
    """"Close" stages the dialog's selected network for the sidebar (via the
    same ``pending_network_label`` mechanism an import uses) without running
    BBN itself -- the main "Run BBN" button is left to do that (see
    ``test_pending_network_selection_runs_bbn_with_the_custom_network``,
    which exercises the actual solve separately: chaining a further
    ``.run()`` here, after "Manage networks" has rendered its "Rename a
    custom network" selectbox once and then closed, hits the same AppTest
    stale-widget quirk documented in the module docstring, just for a
    selectbox rather than a number_input)."""
    from primat.gui import custom_rates, params_form

    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.run(timeout=60)

    _open_manage_dialog(at)
    [radio] = [r for r in at.radio if r.key == "_manage_selected_network"]
    radio.set_value("mynet")
    at.run(timeout=60)
    [close_btn] = [b for b in at.button if b.key == "_btn_manage_close"]
    close_btn.click()
    at.run(timeout=60)
    assert not at.exception

    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    assert network_select.value == "mynet"
    assert at.session_state["_active_custom_network"]["title"] == "mynet"


def test_pending_network_selection_runs_bbn_with_the_custom_network():
    """The sidebar's "network" selectbox resolving to a known custom entry
    (exactly what "Close" stages, see ``test_close_stages_selection_for_the_sidebar``)
    must make "Run BBN" actually solve with it -- seeded directly here (rather
    than by also opening "Manage networks" first) purely to dodge the AppTest
    stale-widget quirk that a further ``.run()`` after it has rendered and
    closed would otherwise hit (see the module docstring)."""
    from primat.gui import custom_rates, params_form

    small_kept = ["n_p__d_g", "d_p__He3_g", "d_d__He3_n", "d_d__t_p", "t_p__a_g",
                  "t_d__a_n", "t_a__Li7_g", "He3_n__t_p", "He3_d__a_p",
                  "He3_a__Be7_g", "Be7_n__Li7_p", "Li7_p__a_a"]
    kept_names = [n for n in small_kept if n != "d_d__t_p"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.session_state["_pending_network_label"] = "mynet"
    at.run(timeout=60)
    assert not at.exception

    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    assert network_select.value == "mynet"

    [run_btn] = [b for b in at.button if b.label == "Run BBN"]
    run_btn.click()
    at.run(timeout=120)
    assert not at.exception
    subheaders = [s.value for s in at.subheader if s.value.endswith(" reactions")]
    assert subheaders[0] == "12 reactions"  # 11 + n__p


def test_amax_auto_unchecked_on_close_but_stays_clickable():
    """Regression guard (CPLAN.md request): "Limit max mass number" must be
    unchecked automatically once a custom network is applied via "Close",
    rather than silently carrying an unrelated previous network's value
    over onto the freshly chosen one -- but it must stay clickable (not
    greyed out), since amax does genuinely filter a custom network's own
    kept-list too (load_network applies it to whatever ``reaction_names``
    it is given, custom or not)."""
    from primat.gui import custom_rates, params_form

    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    [amax_toggle] = [t for t in at.sidebar.checkbox if t.key == "amax_enabled"]
    amax_toggle.set_value(True)
    at.run(timeout=60)
    assert at.session_state["amax_enabled"] is True

    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.session_state["_manage_selected_network"] = "mynet"
    at.run(timeout=60)
    _open_manage_dialog(at)
    [close_btn] = [b for b in at.button if b.key == "_btn_manage_close"]
    close_btn.click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["amax_enabled"] is False
    [amax_toggle] = [t for t in at.sidebar.checkbox if t.key == "amax_enabled"]
    assert not amax_toggle.disabled


def test_amax_auto_unchecked_on_direct_sidebar_switch_to_custom_network():
    """Same guard as above, but for switching the sidebar's own "network"
    dropdown straight to an already-known custom network (no "Manage
    networks" dialog involved) -- the transition is detected purely from
    the network value actually changing, not from a specific button."""
    from primat.gui import custom_rates, params_form

    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    [amax_toggle] = [t for t in at.sidebar.checkbox if t.key == "amax_enabled"]
    amax_toggle.set_value(True)
    at.run(timeout=60)
    assert at.session_state["amax_enabled"] is True

    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.run(timeout=60)
    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    network_select.set_value("mynet")
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["amax_enabled"] is False


# ---------------------------------------------------------------------------
# Remove / rename a custom network
# ---------------------------------------------------------------------------

def test_remove_button_only_offered_for_custom_networks():
    from primat.gui import custom_rates, params_form

    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.run(timeout=60)

    _open_manage_dialog(at)
    assert not [b for b in at.button if b.key == "_manage_remove_small"]
    assert not [b for b in at.button if b.key == "_manage_remove_large"]
    # Remove/Rename only appear once "mynet" itself is the selected entry --
    # toggling the radio is the new "select this row" gesture.
    [radio] = [r for r in at.radio if r.key == "_manage_selected_network"]
    radio.set_value("mynet")
    at.run(timeout=60)
    [remove_btn] = [b for b in at.button if b.key == "_manage_remove_mynet"]
    remove_btn.click()
    at.run(timeout=60)
    assert not at.exception
    assert "mynet" not in at.session_state["_known_custom_networks"]


def test_rename_a_custom_network():
    from primat.gui import custom_rates, params_form

    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.run(timeout=60)

    _open_manage_dialog(at)
    [radio] = [r for r in at.radio if r.key == "_manage_selected_network"]
    radio.set_value("mynet")
    at.run(timeout=60)
    [rename_btn] = [b for b in at.button if b.key == "_manage_rename_apply"]
    rename_btn.click()
    at.run(timeout=60)
    assert at.session_state["_manage_rename_open"] is True

    [new_name] = [t for t in at.text_input if t.key == "_manage_rename_input"]
    assert new_name.value == "mynet"
    new_name.set_value("renamed")
    at.run(timeout=60)
    [confirm_btn] = [b for b in at.button if b.key == "_manage_rename_confirm"]
    assert not confirm_btn.disabled
    confirm_btn.click()
    at.run(timeout=60)
    assert not at.exception
    known = at.session_state["_known_custom_networks"]
    assert "renamed" in known and "mynet" not in known


def test_rename_to_a_reserved_name_is_blocked():
    from primat.gui import custom_rates, params_form

    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), kept_names, {})
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _seed_known_network(at, "mynet", kept_names, custom_network)
    at.run(timeout=60)

    _open_manage_dialog(at)
    [radio] = [r for r in at.radio if r.key == "_manage_selected_network"]
    radio.set_value("mynet")
    at.run(timeout=60)
    [rename_btn] = [b for b in at.button if b.key == "_manage_rename_apply"]
    rename_btn.click()
    at.run(timeout=60)

    [new_name] = [t for t in at.text_input if t.key == "_manage_rename_input"]
    new_name.set_value("small")
    at.run(timeout=60)
    [confirm_btn] = [b for b in at.button if b.key == "_manage_rename_confirm"]
    assert confirm_btn.disabled
    errors = [e.value for e in at.error]
    assert any("built-in network name" in e for e in errors)


# ---------------------------------------------------------------------------
# Download zip is self-contained (every kept reaction's table included)
# ---------------------------------------------------------------------------

def test_download_zip_contains_a_table_for_every_kept_reaction():
    """The dialog's "Download network details" button calls
    ``custom_rates.export_zip`` with the live dialog state; rather than
    fetching the rendered button's bytes (AppTest's ``download_button`` proto
    only carries a "/mock/media/<id>.<ext>" URL into a media-file manager
    that ``AppTest._run`` tears down again before ``.run()`` returns, so
    there is no supported way to read the bytes back out afterwards), apply
    the dialog and call the exact same production function directly with the
    resulting (custom_network, kept_names) snapshot.
    """
    import io
    import zipfile

    from primat.gui import custom_rates, params_form

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    _click_create_this_network(at)
    assert not at.exception

    saved = at.session_state["_known_custom_networks"]["custom"]
    custom_network = saved["custom_network"]
    kept_names = saved["kept"]
    zip_bytes = custom_rates.export_zip(
        params_form._cfg(), custom_network, kept_names, network_filename="custom")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        net_files = [n for n in names if n.startswith("networks/")]
        assert len(net_files) == 1
        net_text = zf.read(net_files[0]).decode()
        kept_lines = [ln for ln in net_text.splitlines() if ln.strip()]
        assert len(kept_lines) == 12
        # Every kept reaction has a "name, name_primat.txt"-style pairing and
        # a corresponding table file -- including shipped-default reactions,
        # not just genuinely customised ones.
        table_files = {n for n in names if n.startswith("tables/")}
        for line in kept_lines:
            assert "," in line, f"{line!r} has no paired table file"
            bare, fname = (p.strip() for p in line.split(",", 1))
            assert f"tables/{bare}/{fname}" in table_files


# ---------------------------------------------------------------------------
# Load a network from a zip via "Manage networks" (FUTURE.md P2 coverage)
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_load_zip_edit_export_roundtrip():
    """A network built with ``custom_rates.kept_to_custom_network`` +
    ``export_zip`` (exactly what the dialog's "Download network details"
    button does, see ``test_download_zip_contains_a_table_for_every_kept_reaction``)
    must load back to an equivalent custom network through the actual
    "Manage networks" -> "Load a network from file" section, and a *further*
    edit of that loaded network must still export correctly -- the full
    build -> export -> load -> edit -> export round trip.

    The "edit" step is driven directly through ``kept_to_custom_network`` /
    ``export_zip`` too, rather than reopening "Create new network" a second
    time in the same ``AppTest`` session (see the module docstring's
    harness-quirk note), which is exactly what the dialog calls under the
    hood anyway (see ``_DialogState.to_custom_network``/``_render_dialog_footer``).
    """
    import io
    import zipfile

    from primat.gui import custom_rates, params_form

    cfg = params_form._cfg()
    small_kept = ["n_p__d_g", "d_p__He3_g", "d_d__He3_n", "d_d__t_p", "t_p__a_g",
                  "t_d__a_n", "t_a__Li7_g", "He3_n__t_p", "He3_d__a_p",
                  "He3_a__Be7_g", "Be7_n__Li7_p", "Li7_p__a_a"]
    kept_names = [n for n in small_kept if n != "d_d__t_p"]
    custom_network = custom_rates.kept_to_custom_network(cfg, kept_names, {})
    zip_bytes = custom_rates.export_zip(
        cfg, custom_network, kept_names, network_filename="roundtrip")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_manage_dialog(at)
    [uploader] = [u for u in at.file_uploader if u.key.startswith("_manage_load_upload")]
    uploader.set_value(("roundtrip.zip", zip_bytes, "application/zip"))
    at.run(timeout=60)
    assert not at.exception

    [add_btn] = [b for b in at.button if b.key == "_manage_load_add"]
    add_btn.click()
    at.run(timeout=60)
    assert not at.exception

    loaded = at.session_state["_known_custom_networks"]["roundtrip"]
    assert sorted(loaded["kept"]) == sorted(kept_names)
    assert "d_d__t_p" not in loaded["kept"]

    # Edit the just-loaded network (drop one more reaction) and export
    # again -- the second export must reflect the edit, not just replay the
    # first one.
    edited_kept = [n for n in kept_names if n != "He3_a__Be7_g"]
    edited_network = custom_rates.kept_to_custom_network(cfg, edited_kept, {})
    zip_bytes2 = custom_rates.export_zip(
        cfg, edited_network, edited_kept, network_filename="roundtrip2")
    with zipfile.ZipFile(io.BytesIO(zip_bytes2)) as zf:
        [net_file] = [n for n in zf.namelist() if n.startswith("networks/")]
        kept_lines = [ln for ln in zf.read(net_file).decode().splitlines() if ln.strip()]
        bare_names = {ln.split(",")[0].strip() for ln in kept_lines}
    assert "He3_a__Be7_g" not in bare_names
    assert "d_d__t_p" not in bare_names
    assert bare_names == set(edited_kept)


def test_loading_a_reserved_network_name_is_blocked():
    """Regression guard (CPLAN.md request): loading a zip whose own network
    name is "small"/"small_parthenope"/"large" must not be allowed to shadow
    the corresponding real, built-in network -- e.g. a "small" exported with
    an amax filter applied, then re-loaded, would otherwise silently make
    every future pick of "small" from the sidebar solve the amax-filtered
    reaction list instead of the true 12-reaction small network."""
    from primat.gui import custom_rates, params_form

    cfg = params_form._cfg()
    kept_names = ["n_p__d_g", "d_p__He3_g"]
    custom_network = custom_rates.kept_to_custom_network(cfg, kept_names, {})
    zip_bytes = custom_rates.export_zip(
        cfg, custom_network, kept_names, network_filename="small")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_manage_dialog(at)
    [uploader] = [u for u in at.file_uploader if u.key.startswith("_manage_load_upload")]
    uploader.set_value(("small.zip", zip_bytes, "application/zip"))
    at.run(timeout=60)
    assert not at.exception

    errors = [e.value for e in at.error]
    assert any("built-in network name" in e for e in errors)
    # AppTest's ``.click()`` bypasses a button's `disabled` state (it has no
    # concept of a browser refusing the click), so the actual guard this
    # test cares about is that the button *is* disabled -- a real browser
    # never lets the click through at all.
    [add_btn] = [b for b in at.button if b.key == "_manage_load_add"]
    assert add_btn.disabled
    # "small" must remain the real, unmodified built-in network.
    assert "small" not in at.session_state["_known_custom_networks"]


# ---------------------------------------------------------------------------
# Invalid uploaded rate table -> clean st.error, no crash (FUTURE.md P2)
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_invalid_uploaded_rate_table_shows_clean_error():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)

    [open_btn] = [b for b in at.button if b.key == "_dialog_add_rate_open_btn"]
    open_btn.click()
    at.run(timeout=60)

    [name_in] = [t for t in at.text_input if t.key == "_dialog_add_name"]
    name_in.set_value("Li6_d__a_a")
    at.run(timeout=60)
    assert not at.exception

    [uploader] = [u for u in at.file_uploader if u.key == "_dialog_add_table"]
    uploader.set_value(("bad.txt", b"this is not a rate table\njust some text\n", "text/plain"))
    at.run(timeout=60)
    assert not at.exception

    [add_btn] = [b for b in at.button if b.key == "_dialog_add_submit"]
    assert not add_btn.disabled
    add_btn.click()
    at.run(timeout=60)
    assert not at.exception

    errors = [e.value for e in at.error]
    assert any("Rate table" in e for e in errors), errors
    # The malformed upload must not have been accepted.
    assert "Li6_d__a_a" not in at.session_state["_dialog_added"]
    assert at.session_state["_dialog_add_rate_open"] is True


# ---------------------------------------------------------------------------
# Export/import round trip of an *unmodified* network must look unmodified:
# Source labels keep their original ref= provenance, never "custom upload",
# since no rate table was actually uploaded by the user (CPLAN.md request).
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_unmodified_network_roundtrip_keeps_original_source_labels():
    """Selecting an existing network (e.g. ``large`` with ``amax=8``),
    exporting it as a zip and re-importing it as a custom network (with no
    edits in between) must reproduce the same per-reaction "Source" labels
    as the original network -- i.e. each reaction's ``ref=`` provenance, not
    the generic "custom rate" placeholder, which must only ever appear for
    a reaction whose table really was uploaded/edited by the user.
    """
    import io

    from primat.config import PRIMATConfig
    from primat.network_data import UpdateNuclearRates
    from primat.gui import custom_rates

    cfg = PRIMATConfig({"network": "large", "amax": 8})
    original = UpdateNuclearRates(cfg).describe_reactions()
    names = [name for name, _eq, _src, _file in original]
    source_by_name = {name: src for name, _eq, src, _file in original}

    zip_bytes = custom_rates.export_zip(
        cfg, {"removed": [], "replaced": {}, "added": {}}, names,
        network_filename="roundtrip")
    imported = custom_rates.import_zip(io.BytesIO(zip_bytes))
    custom_network = custom_rates.kept_to_custom_network(
        cfg, imported["kept"], imported["replaced"],
        decay_overrides=imported["decay_overrides"],
        filenames=imported.get("filenames"))

    roundtripped = UpdateNuclearRates(cfg, custom_network=custom_network).describe_reactions()
    for name, _eq, src, _file in roundtripped:
        assert src == source_by_name[name], (
            f"{name}: source changed from {source_by_name[name]!r} to {src!r} "
            "after an unmodified export/import round trip"
        )
        assert src != "custom rate"


# ---------------------------------------------------------------------------
# Reopening a previously built/imported custom network in "Create new
# network" must not mislabel its unmodified, shipped-default tables as
# "_custom" (CPLAN.md request)
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_reopening_a_round_tripped_network_keeps_shipped_filenames():
    import io

    from primat.gui import custom_rates, params_form

    cfg = params_form._cfg()
    kept_names = ["n_p__d_g", "d_p__He3_g", "d_d__He3_n"]
    custom_network = custom_rates.kept_to_custom_network(cfg, kept_names, {})
    zip_bytes = custom_rates.export_zip(
        cfg, custom_network, kept_names, network_filename="roundtrip")
    imported = custom_rates.import_zip(io.BytesIO(zip_bytes))
    reimported_network = custom_rates.kept_to_custom_network(
        cfg, imported["kept"], imported["replaced"],
        decay_overrides=imported["decay_overrides"],
        filenames=imported.get("filenames"))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.session_state["_known_custom_networks"] = {
        "roundtrip": {"kept": list(kept_names), "tables": dict(imported["replaced"]),
                     "custom_network": reimported_network},
    }
    at.run(timeout=60)

    _open_manage_dialog(at)
    [radio] = [r for r in at.radio if r.key == "_manage_selected_network"]
    radio.set_value("roundtrip")
    at.run(timeout=60)
    [create_btn] = [b for b in at.button if b.key == "_btn_create_new_network"]
    create_btn.click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["_dialog_base_network"] == "roundtrip"

    table_choice = at.session_state["_dialog_table_choice"]
    for name in kept_names:
        # None of these were ever genuinely customised -- every one must
        # keep reading as the real shipped file, never "<name>_custom...".
        assert table_choice.get(name, "").endswith("_primat.txt"), (
            f"{name}: expected a shipped '_primat.txt' table, got "
            f"{table_choice.get(name)!r}"
        )
