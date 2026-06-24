# -*- coding: utf-8 -*-
"""
primat.gui.session_keys
==========================

Single source of truth for every ``st.session_state`` key the GUI touches.

Why this exists
----------------
``st.session_state`` is a plain ``dict[str, Any]`` shared across the whole
Streamlit script run; a key typo'd in one place (or renamed in one place but
not another) fails silently -- ``st.session_state.get("typo")`` just returns
``None``/the fallback, it never raises. With ~50 keys read and written across
``params_form.py`` (the custom-network dialog), ``app.py`` and ``panels.py``,
that risk is real. Centralising every key as a named constant here turns a
typo into a ``NameError``/``AttributeError`` at import time instead of a
silent no-op at runtime, and makes "what touches this key" a single grep for
the constant name instead of a grep for a string that may also appear as a
substring of something unrelated.

Streamlit widgets persist their value under exactly the string passed as
``key=``, so the *values* of the constants below are part of the GUI's
behaviour (and, transitively, of ``tests/test_gui*.py``, which assert against
several of them directly via ``AppTest``'s ``session_state``/widget
``.key``). Do not change a value without checking those tests.

Most keys are static (module-level string constants). A handful are
*per-reaction* or *per-category*, parameterised by the dialog's reaction name
and/or its remount generation counter (see :data:`SessionKeys.dialog_gen`'s
docstring on the file's own ``_DialogState.bump_gen``) -- those are exposed as
``staticmethod`` key-builders instead of constants.
"""


class SessionKeys:
    """Namespace of ``st.session_state`` key constants (never instantiated)."""

    # ---- Cross-module contract (app.py / panels.py / params_form.py) ------
    # The result of the sidebar form / custom-network dialog, consumed by
    # app.py's run/solve step and by panels.py's Reactions tab.
    params = "params"
    quick_mc = "quick_mc"
    mc_samples = "mc_samples"
    active_custom_network = "_active_custom_network"
    run_custom_network_dict = "run_custom_network_dict"
    network = "network"

    # ---- Sidebar form (params_form.render_sidebar_form) --------------------
    amax_enabled = "amax_enabled"
    amax_value = "amax_value"
    quick_mc_uncertainty = "quick_mc_uncertainty"
    quick_mc_samples = "quick_mc_samples"
    pending_network_label = "_pending_network_label"
    amax_prev_network = "_amax_prev_network"
    known_custom_networks = "_known_custom_networks"

    # ---- "Create custom network" dialog: show/hide -------------------------
    show_custom_dialog = "_show_custom_dialog"

    # ---- "Manage networks" dialog (lists/removes/loads/renames, and is the
    # sole entry point into the "Create custom network" dialog) -------------
    show_manage_dialog = "_show_manage_dialog"
    btn_manage_networks = "_btn_manage_networks"
    manage_selected_network = "_manage_selected_network"
    manage_load_upload_gen = "_manage_load_upload_gen"
    manage_load_title = "_manage_load_title"
    manage_load_add = "_manage_load_add"

    @staticmethod
    def manage_load_upload(gen):
        # Embeds a generation counter so bumping it remounts the
        # file_uploader with a fresh empty widget (Streamlit does not let a
        # file_uploader's selection be cleared by writing to session_state).
        return f"_manage_load_upload_{gen}"
    manage_rename_open = "_manage_rename_open"
    manage_rename_input = "_manage_rename_input"
    manage_rename_apply = "_manage_rename_apply"
    manage_rename_confirm = "_manage_rename_confirm"
    btn_create_new_network = "_btn_create_new_network"
    btn_manage_close = "_btn_manage_close"
    last_created_network = "_last_created_network"

    @staticmethod
    def manage_remove_btn(name):
        return f"_manage_remove_{name}"

    @staticmethod
    def manage_modify_btn(name):
        return f"_manage_modify_{name}"

    @staticmethod
    def manage_download_btn(name):
        return f"_manage_download_{name}"

    # ---- "Create custom network" dialog: per-session, not per-reaction -----
    dialog_title = "_dialog_title"
    dialog_gen = "_dialog_gen"
    dialog_decay_override = "_dialog_decay_override"
    dialog_pending_open = "_dialog_pending_open"
    dialog_base_network = "_dialog_base_network"
    dialog_prev_base_network = "_dialog_prev_base_network"
    dialog_amax_enabled = "_dialog_amax_enabled"
    dialog_amax_value = "_dialog_amax_value"
    dialog_signature = "_dialog_signature"
    dialog_added = "_dialog_added"
    dialog_keep = "_dialog_keep"
    dialog_table_choice = "_dialog_table_choice"
    dialog_uploaded_tables = "_dialog_uploaded_tables"
    dialog_add_rate_open = "_dialog_add_rate_open"
    dialog_add_rate_open_btn = "_dialog_add_rate_open_btn"
    dialog_add_name = "_dialog_add_name"
    dialog_add_table = "_dialog_add_table"
    dialog_add_submit = "_dialog_add_submit"
    dialog_add_cancel = "_dialog_add_cancel"
    dialog_download = "_dialog_download"
    dialog_apply = "_dialog_apply"
    dialog_selall_decay = "_dialog_selall_decay"
    dialog_deselall_decay = "_dialog_deselall_decay"
    dialog_expander_decay = "_dialog_expander_decay"

    # ---- Per-reaction-name / per-category widget keys (dynamic) ------------
    # Embeds `gen` (see `dialog_gen` above) so Select-all/Deselect-all and a
    # base-network change -- which mutate the backing dicts (`dialog_keep`
    # etc.) directly rather than through the widget -- actually take visible
    # effect: a Streamlit widget keeps its own persisted value forever once a
    # given key has been used, regardless of any later `value=`/`index=`, so
    # each remount needs a fresh key. See `_DialogState.bump_gen` in
    # `params_form.py` for the full explanation; every other docstring in
    # this module should point here rather than re-explain it.

    @staticmethod
    def dialog_keep_widget(gen, name):
        return f"_dialog_keep_{gen}_{name}"

    @staticmethod
    def dialog_table_widget(gen, name):
        return f"_dialog_table_{gen}_{name}"

    @staticmethod
    def dialog_addtable_widget(gen, name):
        return f"_dialog_addtable_{gen}_{name}"

    @staticmethod
    def dialog_upload_widget(name):
        return f"_dialog_upload_{name}"

    @staticmethod
    def dialog_show_uploader(name):
        return f"_dialog_show_uploader_{name}"

    @staticmethod
    def dialog_decay_rate_widget(gen, name):
        return f"_dialog_decay_rate_{gen}_{name}"

    @staticmethod
    def dialog_selall(cat):
        return f"_dialog_selall_{cat}"

    @staticmethod
    def dialog_deselall(cat):
        return f"_dialog_deselall_{cat}"

    @staticmethod
    def dialog_expander_cat(cat):
        return f"_dialog_expander_cat_{cat}"
