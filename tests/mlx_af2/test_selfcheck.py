from bindcraft import selfcheck

def test_selfcheck_asks_for_mlx_only_on_the_mlx_backend(monkeypatch):
    monkeypatch.setattr(selfcheck, 'module_present', lambda name: name != 'mlx.core')
    monkeypatch.delenv('BINDCRAFT_BACKEND', raising=False)
    assert 'mlx.core' not in selfcheck.missing_modules()
    monkeypatch.setenv('BINDCRAFT_BACKEND', 'mlx')
    assert 'mlx.core' in selfcheck.missing_modules()
    monkeypatch.delenv('BINDCRAFT_BACKEND')
    assert 'mlx.core' in selfcheck.missing_modules('mlx')
