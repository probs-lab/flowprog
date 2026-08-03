
def pytest_sessionstart():
    # jax/numpyro are an optional extra. Only configure jax when it is
    # installed so the rest of the suite can run without it (the numpyro tests
    # skip themselves via pytest.importorskip).
    try:
        import jax
    except ImportError:
        return
    jax.config.update("jax_enable_x64", True)
