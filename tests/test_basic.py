import pylibjxl

def test_version():
    v = pylibjxl.version()
    assert "major" in v
    assert "minor" in v
    assert "patch" in v
    print(f"libjxl version: {v['major']}.{v['minor']}.{v['patch']}")

def test_decoder_version():
    v = pylibjxl.decoder_version()
    assert v > 0
    print(f"Decoder version: {v}")

def test_encoder_version():
    v = pylibjxl.encoder_version()
    assert v > 0
    print(f"Encoder version: {v}")
