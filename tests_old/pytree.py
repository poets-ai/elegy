import jax
import jax.numpy as jnp

import elegy as eg


class TestPyTree:
    def test_pytree(self):
        class Tree(eg.PytreeObject):
            node: int
            static: int = eg.field(node=False)

            def __init__(self, node: int, static: int):
                self.node = node
                self.static = static

        tree = Tree(1, 2)

        tree_leaves = jax.tree_leaves(tree)

        assert len(tree_leaves) == 1
