import jax
import jax.numpy as jnp

import elegy as eg


class TestPyTree:
    def test_pytree(self):
        class Tree(eg.PytreeObject):
            node: int
            static: int = eg.field(pytree_node=False)

            def __init__(self, node: int, static: int):
                self.node = node
                self.static = static

        tree: Tree = Tree(1, 2)

        tree_leaves = jax.tree_leaves(tree)

        assert tree_leaves == [1]

        tree = jax.tree_map(lambda x: x + 5, tree)

        assert tree.node == 6

    def test_default_values(self):
        class Tree(eg.PytreeObject):
            node = 5
            static = eg.field(10, pytree_node=False)

        tree: Tree = Tree()

        assert tree.node == 5
        assert tree.static == 10

        leaves = jax.tree_leaves(tree)

        assert leaves == [5]
        assert tree.static == 10

        tree = jax.tree_map(lambda x: x - 5, tree)

        assert tree.node == 0
        assert tree.static == 10
