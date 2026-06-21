"""
Tests for the GUI's "Create custom network"/"Import custom network" popups
(``pyprimat.gui.params_form``, see CUSTOMPOPUP.md §5-§8).

These drive the actual popup workflow end to end with Streamlit's ``AppTest``
harness (no browser needed), mimicking the user flows a person would actually
click through: opening/switching between the two dialogs, toggling reactions
on/off (individually and via Select all/Deselect all), switching the base
network, editing a decay rate, adding a brand-new reaction, and applying the
result to a real ``PyPR`` solve. Each is a regression guard for a bug found
and fixed by hand-testing the popup:

* the two dialogs not being mutually exclusive after one was dismissed via
  its own close button (``_render_custom_network_buttons``);
* Select all/Deselect all having no visible effect, and switching the base
  network not refreshing the category list, both caused by Streamlit widgets
  silently keeping their own stale state once a key has been used once
  (``_bump_dialog_gen``);
* reactions above the dialog's own ``amax`` silently staying in the solved
  network because ``removed`` was computed against the *filtered* view
  instead of the full large-network list (``_dialog_to_custom_network``);
* a decay reaction showing a misleading "(no table)" instead of its own
  editable rate (``_render_decay_category``/``_render_decay_row``);
* a brand-new "Add new rate" reaction showing "(no table)" in its own
  category despite having an uploaded table (``_render_reaction_row``).

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

APP_PATH = "pyprimat/gui/app.py"

_AC2024_DIR = os.path.join(os.path.dirname(__file__), "..", "pyprimat",
                           "rates", "nuclear", "data")
_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/nuclear/data not generated",
)


def _open_create_dialog(at):
    [btn] = [b for b in at.sidebar.button if b.label == "Create/modify network"]
    btn.click()
    at.run(timeout=60)
    return at


def _open_import_dialog(at):
    [btn] = [b for b in at.sidebar.button if b.label == "Import custom network"]
    btn.click()
    at.run(timeout=60)
    return at


def _toggle_keep(at, bare_name, value):
    """Flip a reaction row's "keep" toggle, regardless of its current `_dialog_gen`
    suffix (the key embeds a generation counter that changes whenever the
    backing state is rebuilt -- see ``_bump_dialog_gen``)."""
    [t] = [t for t in at.toggle if t.key and t.key.endswith(f"_{bare_name}")
          and t.key.startswith("_dialog_keep_")]
    t.set_value(value)
    at.run(timeout=60)
    return at


def _click_apply(at):
    [btn] = [b for b in at.button if b.key == "_dialog_apply"]
    btn.click()
    at.run(timeout=120)
    return at


# ---------------------------------------------------------------------------
# Buttons: stacked, mutually exclusive
# ---------------------------------------------------------------------------

def test_buttons_are_stacked_and_mutually_exclusive():
    """The two buttons render as separate full-width rows (not side by side),
    and opening one always closes the other, even after the other was
    dismissed via its own close button (which leaves its show-flag stuck
    True -- the bug report this guards against)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    labels = [b.label for b in at.sidebar.button]
    assert labels[:2] == ["Import custom network", "Create/modify network"]

    _open_import_dialog(at)
    assert at.session_state["_show_import_dialog"] is True
    assert at.session_state["_show_custom_dialog"] is False

    # Simulate "dismissed via the dialog's own close (x) button": our
    # show-flag has no way to observe that, so it would still be True here in
    # the real app. Clicking "Create custom network" must still end up with
    # exactly one dialog open.
    _open_create_dialog(at)
    assert not at.exception
    assert at.session_state["_show_custom_dialog"] is True
    assert at.session_state["_show_import_dialog"] is False


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
# amax correctness: nothing above the dialog's amax leaks into the solve
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_amax_filtered_reactions_are_actually_removed_on_apply():
    """Regression guard: reactions above the dialog's own amax must end up in
    custom_network["removed"], not silently survive because they were never
    shown (and so never explicitly toggled off)."""
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

    _click_apply(at)
    assert not at.exception

    import json
    custom_network = json.loads(at.session_state["params"]["custom_network"])
    from pyprimat.network_data import reaction_category
    # This is the actual regression this guards: reactions the dialog never
    # even showed (everything above amax=8) must still land in "removed",
    # not be silently kept just because the user never saw a toggle for them.
    above_amax = [n for n in custom_network["removed"]
                 if n not in custom_network.get("added", {})
                 and reaction_category(n) > 8]
    assert above_amax, "no above-amax reaction found in 'removed' -- amax leak regressed"
    # large+amax8 == the old "medium" network's 68 reactions (67 + n__p);
    # describe_reactions() includes the prepended n__p.
    active = at.session_state["_active_custom_network"]
    assert len(active["kept"]) == 67


# ---------------------------------------------------------------------------
# Removing a single reaction
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_removing_one_reaction_and_applying_drops_it_from_the_solve():
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

    _click_apply(at)
    assert not at.exception

    import json
    custom_network = json.loads(at.session_state["params"]["custom_network"])
    assert "d_d__t_p" in custom_network["removed"]

    subheaders = [s.value for s in at.subheader if s.value.endswith(" reactions")]
    assert subheaders, "Reactions tab subheader not found"
    # 67 reactions at amax=8 minus 1 removed, + the prepended n__p = 67.
    assert subheaders[0] == "67 reactions"


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
# Apply and run BBN end to end
# ---------------------------------------------------------------------------

def test_apply_and_run_bbn_solves_with_the_customised_network():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    _toggle_keep(at, "d_d__t_p", False)
    _click_apply(at)
    assert not at.exception

    subheaders = [s.value for s in at.subheader if s.value.endswith(" reactions")]
    assert subheaders[0] == "12 reactions"  # 11 + n__p

    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    assert network_select.value == "custom"


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

    from pyprimat.gui import custom_rates, params_form

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    _open_create_dialog(at)
    _click_apply(at)
    assert not at.exception

    custom_network = at.session_state["run_custom_network_dict"]
    kept_names = at.session_state["_active_custom_network"]["kept"]
    zip_bytes = custom_rates.export_zip(
        params_form._cfg(), custom_network, kept_names, network_filename="custom")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        net_files = [n for n in names if n.startswith("networks/")]
        assert len(net_files) == 1
        net_text = zf.read(net_files[0]).decode()
        kept_lines = [ln for ln in net_text.splitlines() if ln.strip()]
        assert len(kept_lines) == 12
        # Every kept reaction has a "name, name_custom.txt" pairing and a
        # corresponding table file -- including shipped-default reactions,
        # not just genuinely customised ones.
        table_files = {n for n in names if n.startswith("tables/")}
        for line in kept_lines:
            assert "," in line, f"{line!r} has no paired table file"
            bare, fname = (p.strip() for p in line.split(",", 1))
            assert f"tables/{bare}/{fname}" in table_files


# ---------------------------------------------------------------------------
# Import -> edit -> export round trip (FUTURE.md P2 coverage)
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_import_zip_edit_export_roundtrip():
    """A network built with ``custom_rates.kept_to_custom_network`` +
    ``export_zip`` (exactly what the dialog's "Download network details"
    button does, see ``test_download_zip_contains_a_table_for_every_kept_reaction``)
    must import back to an equivalent custom network through the actual
    "Import custom network" dialog, and a *further* edit of that imported
    network must still export correctly -- the full
    build -> export -> import -> edit -> export round trip CUSTOMPOPUP.md
    promises.

    The "edit" step is driven directly through ``kept_to_custom_network`` /
    ``export_zip`` too, rather than by reopening the "Create/modify network"
    dialog a second time in the same ``AppTest`` session: this Streamlit
    version's ``AppTest`` cannot re-render a second ``@st.dialog`` after an
    earlier one closed itself via ``st.rerun()`` (a harness limitation, not a
    real app bug -- reopening the dialog after applying/importing works fine
    in an actual browser session), so this test exercises the dialog's own
    apply path only once (the import) and the underlying data functions for
    the edit, which is exactly what the dialog calls under the hood
    (see ``_DialogState.to_custom_network``/``_render_dialog_footer``).
    """
    import io
    import zipfile

    from pyprimat.gui import custom_rates, params_form

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
    _open_import_dialog(at)
    [uploader] = [u for u in at.file_uploader if u.key == "_import_dialog_upload"]
    uploader.set_value(("roundtrip.zip", zip_bytes, "application/zip"))
    at.run(timeout=60)
    assert not at.exception

    imported = at.session_state["_known_custom_networks"]["roundtrip"]
    assert sorted(imported["kept"]) == sorted(kept_names)
    assert "d_d__t_p" not in imported["kept"]
    assert at.session_state["_active_custom_network"]["title"] == "roundtrip"

    # Edit the just-imported network (drop one more reaction) and export
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


# ---------------------------------------------------------------------------
# "Limit A" shrinks a *custom* base network's kept set too (FUTURE.md P2)
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_amax_filter_applied_to_custom_base_network():
    """Regression guard: enabling/tightening "Limit A" while a previously
    built/imported *custom* network is the dialog's own base must shrink its
    kept set (the `base_network in known` branch of `_DialogState.reset`),
    not just leave every one of its reactions "kept" regardless of amax.

    The custom network is seeded directly into ``_known_custom_networks``
    (rather than built live through "Create/modify network" -> Apply) so the
    dialog is opened only once in this ``AppTest`` session -- see
    ``test_import_zip_edit_export_roundtrip``'s docstring for why a *second*
    dialog open after an earlier ``st.rerun()`` is unreliable under this
    Streamlit version's ``AppTest`` harness.
    """
    from pyprimat.gui import custom_rates, params_form

    small_kept = ["n_p__d_g", "d_p__He3_g", "d_d__He3_n", "d_d__t_p", "t_p__a_g",
                  "t_d__a_n", "t_a__Li7_g", "He3_n__t_p", "He3_d__a_p",
                  "He3_a__Be7_g", "Be7_n__Li7_p", "Li7_p__a_a"]
    custom_network = custom_rates.kept_to_custom_network(
        params_form._cfg(), small_kept, {})
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.session_state["_known_custom_networks"] = {
        "mynet": {"kept": list(small_kept), "tables": {}, "custom_network": custom_network},
    }
    at.session_state["_pending_network_label"] = "mynet"
    at.run(timeout=60)
    assert not at.exception

    _open_create_dialog(at)
    assert not at.exception
    assert at.session_state["_dialog_base_network"] == "mynet"
    assert sum(1 for v in at.session_state["_dialog_keep"].values() if v) == 12

    [amax_value] = [n for n in at.number_input if n.key == "_dialog_amax_value"]
    amax_value.set_value(2)
    at.run(timeout=60)
    assert not at.exception

    kept = sorted(n for n, v in at.session_state["_dialog_keep"].items() if v)
    # Of the small network's 12 reactions, only n_p__d_g has A <= 2.
    assert kept == ["n_p__d_g"]


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
