import random

def evaluate_travel_distance(clients):
  # todo
  return random.random()

def evaluate_overlap_loss(clients):
  min_x = min(c.x for c in clients)
  max_x = max(c.x for c in clients)
  min_y = min(c.y for c in clients)
  max_y = max(c.y for c in clients)
  area = (max_x - min_x) * (max_y - min_y)
  return area

def evaluate(clients, labels):
  assert len(clients) == len(labels)
  num_labels = max(labels) + 1
  loss = 0.0
  max_dist = 0.0
  for label in range(num_labels):
    clients = [c for c,l in zip(clients,labels) if l == label]
    if not clients:
      continue
    dist = evaluate_travel_distance(clients)
    max_dist = max(max_dist, dist)
    loss += dist / num_labels
    loss += evaluate_overlap_loss(clients)
  loss += max_dist
  return loss

def search(clients, num_employees):
  best_labels = [0] * len(clients)
  best_loss = evaluate(clients, best_labels)
  while True:
    new_labels = best_labels.copy()
    new_labels[random.randint(0, len(clients)-1)] = random.randint(0, num_employees-1)
    new_loss = evaluate(clients, new_labels)
    if new_loss < best_loss:
      best_labels = new_labels
      best_loss = new_loss
  return best_labels