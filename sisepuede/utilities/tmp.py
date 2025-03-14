
def depth_node(
    node: Node,
) -> int:
    d = 0
    while node.parent is not None:
        node = node.parent
        d += 1

    return d




def height_node(
    node: Node,
) -> int:
    if node is None: return -1  # subtract one from the 1 + below
    height = 1 + max(height_node(node.left), height_node(node.right))
    return height


def size_tree(
    node: Node,
) -> int:
    if node is None: return 0  # subtract one from the 1 + below
    size = 1 + size_tree(node.left) + size_tree(node.right)
    return size



def breadth_first_traverse(
    root: Node, 
):

    queue = LinkedList() # list of nodes

    if root is not None:
        queue.push(root)
    
    while not queue.is_empty():
        node = queue.pop()
        if node.left is not None: queue.push(node.left)
        if node.right is not None: queue.push(node.right)
    
    return None


def bfs_graph(
    graph: Graph, 
):

    seen = list[0 for x in nv(graph)] # bool
    queue = SingleLinkedList()
    v = graph.vertices

    # push the first vertex (assume integers)
    queue.push(v[0])
    seen[v[0]] = 1

    while not queue.is_empty():
        i = queue.pop()
        for j in graph.neighbors(i):
            if not seen[j]:
                seen[j] = 1
                queue.push(j)
    
    return None





def dfs_graph_recursive(
    graph: Graph,
    v: int,
    visited: set,
):
    # update the visited set
    visited.add(v)

    # use recursion on adjacent vertices
    for j in graph.neighbors(v):
        if j not in visited:
            dfs_graph_neighbors(j, visited)


def dfs_graph(
    graph: Graph, 
    v: int
):
    # Create a set to store visited vertices
    visited = set()
    dfs_graph_recursive(v, visited)