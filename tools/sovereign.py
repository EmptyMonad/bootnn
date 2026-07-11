#!/usr/bin/env python3
"""
Self-sovereign, registry-free authorship (S2 identity decision).

Authorship anchors to the append-only log: precedence is entry order,
integrity is the hash chain — identity-free by default. This module is
the OPTIONAL layer: a Merkle–Lamport identity a contributor m