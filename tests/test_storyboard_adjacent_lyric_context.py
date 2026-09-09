from pathlib import Path


UI_SOURCE = (Path(__file__).parents[1] / "web" / "VRGDG_StoryboardBuilderUI.js").read_text(encoding="utf-8")


def test_adjacent_lyric_context_is_optional_and_saved():
    assert "Send last and next lyric line for context" in UI_SOURCE
    assert "send_adjacent_lyric_context: Boolean(state.sendAdjacentLyricContext)" in UI_SOURCE
    assert "state.sendAdjacentLyricContext = adjacentLyricContextInput.checked;" in UI_SOURCE


def test_adjacent_lines_are_added_to_scene_beat_requests_only_when_enabled():
    assert "if (!state.sendAdjacentLyricContext)" in UI_SOURCE
    assert 'previous_lyrics: previousLyrics' in UI_SOURCE
    assert 'next_lyrics: nextLyrics' in UI_SOURCE
