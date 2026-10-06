"""Simple tree structure for mapping inventories. Includes summation groups by

"""
from typing import *


class InventoryNode:
    """Node in an inventory tree with arbitrary numbers of branches at each node
        --used to map CRT rows to entries. Includes the following properties:

        * children : List[InventoryNode] 
            All children nodes associated with the current node. 

        * emissions : Dict[str, Union[float, int]]
            Dictionary of emissions associated with the inventory node. 
                Emissions are drawn from the table in the CRT defining it.

        * memo : Bool 
            Bool definining whether or not the node is associated with a memo
                item. Used in summations.

        * name : str
            Name associated with the code. In CRT, this is the IPCC inventory
                code (e.g., 1.A.1.a.i)
        
        * parent : Union[InventoryNode, str] 
            Parent node. If None, assumed to be Root.
    """

    def __init__(self,
        name: str,
        dict_emissions: dict = {},
        memo: Union[bool, None] = None,
    ) -> None:
        """
        """
        
        # initialize emissions, name, and children
        self.children = []
        self.emissions = dict_emissions
        self.memo = memo
        self.name = name
        self.parent = None
            
        return None



    def __str__(self,
    ) -> str:
        n_emissions = len(self.emissions)
        parent = (
            self.parent.name
            if isinstance(self.parent, InventoryNode)
            else "Root"
        )
        
        out = f"{self.name} - InventoryNode with {n_emissions} emissions\nParent: {parent}"

        return out



    def __repr__(self,
    ) -> str:
        return self.__str__()

        

    def append(self,
        obj: 'InventoryNode'
    ) -> None:
        """Append a node
        """
        self.children.append(obj)

        return None
    



##################
#    FUNCTION    #
##################

def describe_tree_dfs(
    node: 'InventoryNode',
) -> Union[str, None]:
    """Print each node of the tree using dfs
    """

    print(node.__str__())
    print("|")

    # recursive search on children
    for node in node.children:
        describe_tree_dfs(node, )

    return None



def get_inventory_tree_sum_dfs(
    tree: 'InventoryNode',
    include_memo: bool = False,
) -> Union[str, None]:
    """Using an inventory tree, get the sum of emissions across components.
    """

    dict_sum = {}

    # case if reach the bottom
    if len(tree.children) == 0:
        if tree.memo:
            out = (
                {}
                if not include_memo
                else tree.emissions
            )

            return out

        return tree.emissions
        

    ##  OTHERWISE, RECURSE
    
    for node in tree.children:
        if node.memo and not include_memo: continue
            
        dict_out = get_inventory_tree_sum_dfs(
            node,
            include_memo = include_memo,
        )

        for k, v in dict_out.items():
            if k in dict_sum.keys():
                dict_sum[k] += v

            else:
                dict_sum.update({k: v, })

    return dict_sum



def get_node_dfs(
    node: 'InventoryNode',
    name: str,
) -> Union[str, None]:
    """Get a node using DFS
    """
    if node.name == name:
        return node
    
    # recursive search on children
    for node in node.children:
        out = get_node_dfs(
            node,
            name,
        )

        if isinstance(out, InventoryNode):
            return out

    return None


            
def get_node_value_dfs(
    node: 'InventoryNode',
    name: str,
) -> Union[str, None]:
    """Get value from a node with name 
    """
    out = get_node_dfs(node, name, )
    if out is None:
        return None

    return out.emissions


