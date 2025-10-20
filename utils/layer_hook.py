class FeatureExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {}
        self._register_hooks()

    def _get_layer(self, name):
        module = dict([*self.model.named_modules()])[name]
        return module

    def _register_hooks(self):
        for name in self.layers:
            layer = self._get_layer(name)
            layer.register_forward_hook(self.save_output_hook(name))

    def save_output_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def get_features(self):
        return self.features