from types import SimpleNamespace

from app.services import parse_transcription


def test_parse_transcription_with_speakers_and_words():
    response = SimpleNamespace(parts=[
        SimpleNamespace(
            text="Hello there. ",
            audio_transcription=SimpleNamespace(
                text=None,
                speaker_label="SPEAKER_1",
                words=[SimpleNamespace(word="Hello", start_offset="0s", end_offset="0.3s")],
            ),
        ),
        SimpleNamespace(
            text="General Kenobi.",
            audio_transcription=SimpleNamespace(text=None, speaker_label="SPEAKER_2", words=[]),
        ),
    ])

    parsed = parse_transcription(response, 321)
    assert parsed["text"] == "Hello there. General Kenobi."
    assert parsed["segments"][1]["speaker"] == "SPEAKER_2"
    assert parsed["words"][0] == {"word": "Hello", "start": "0s", "end": "0.3s"}
    assert parsed["elapsed_ms"] == 321


def test_parse_transcription_falls_back_to_response_text():
    parsed = parse_transcription(SimpleNamespace(parts=[], text="Fallback"), 1)
    assert parsed["text"] == "Fallback"
