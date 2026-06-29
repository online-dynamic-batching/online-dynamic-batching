# Security

Please report security issues privately to the maintainers before opening a
public issue. Include the affected version, reproduction steps, and the expected
impact.

ODB does not execute model code on its own; it wraps a user-provided PyTorch
DataLoader and collate function. Treat datasets, transforms, and collators as
trusted code in your training environment.
