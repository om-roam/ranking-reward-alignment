from coral.training.reward.process import batch_process
from coral.training.reward.labels import coral_targets
from coral.utils.device import move_to_device

def get_inputs(batch, tokenizer, num_classes=4, device=None):
    queries, docs, labels, candidate_mask = batch_process(batch)
    coral_labels = coral_targets(labels, num_classes=num_classes).to(device)
    inputs = tokenizer.batch_encode(queries, docs)

    labels, candidate_mask = labels.to(device), candidate_mask.to(device)
    inputs = move_to_device(inputs, device=device)
    return inputs, coral_labels, candidate_mask