def listwise_collate_fn(batch):
    queries = [item["query"] for item in batch]
    docs = [item["docs"] for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "queries": queries,   
        "docs": docs,         
        "labels": labels,     
    }
