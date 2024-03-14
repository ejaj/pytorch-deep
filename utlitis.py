class Dam(object):
    def __init__(self):
        self.val_step_fn = self._make_val_step_fn()

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            print(x, y)

        return perform_val_step_fn


dam_instance = Dam()
dam_instance.val_step_fn("Hello", "World")
