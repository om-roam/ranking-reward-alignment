def get_transformer_layers(model):
    if hasattr(model, "backbone"):
        model = model.backbone

    # BERT / RoBERTa / DeBERTa
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer

    # LLaMA / Qwen / Mistral
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # Other Qwen variants
    if hasattr(model, "layers"):
        return model.layers

    raise ValueError(
        f"Unsupported model architecture: {type(model)}"
    )

def freeze_layers_encoder(model, n_layers=4):
    for p in model.encoder.parameters():
        p.requires_grad = False

    for layer in get_transformer_layers(model.encoder)[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True