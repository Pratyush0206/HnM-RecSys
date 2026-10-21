from hmrec import config


def average_precision_at_k(actual, predicted, k=config.K):
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    seen = set()
    hits = 0
    score = 0.0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual_set and item not in seen:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(len(actual_set), k)


def map_at_k(actual_by_customer, predicted_by_customer, k=config.K):
    if not actual_by_customer:
        return 0.0
    total = 0.0
    for customer, actual in actual_by_customer.items():
        predicted = predicted_by_customer.get(customer, [])
        total += average_precision_at_k(actual, predicted, k)
    return total / len(actual_by_customer)