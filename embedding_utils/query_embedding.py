from string import digits
import re
import random
class TreeNode:
    def __init__(self, node_representation) -> None:
        self.node_representation = node_representation
        self.children = []
        self.sibling = None
    
def random_walk(root, steps):
    current_node = root
    if current_node.node_representation == "STOP":
        path = []
    else:
        path = [current_node.node_representation]
    while len(path) < steps:
        if not current_node.children and not current_node.sibling:
            break
        neighbors = [child for child in current_node.children]
        sibling = current_node.sibling
        if sibling:
            neighbors.append(sibling)
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        if not next_node:
            break
        if next_node.node_representation != "STOP":
            path.append(next_node.node_representation)
        current_node = next_node
    return path

def generate_walks(root, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        walks.append(random_walk(root, walk_length))
    return walks



INTERESTING_OPERATORS = ["Seq Scan",
                              "Hash Join",
                              "Nested Loop",
                              "CTE Scan",
                              "Index Only Scan",
                              "Index Scan",
                              "Merge Join",
                              "Sort"]
def parse_plan(plan):
    node_type = plan["Node Type"]
    if node_type in INTERESTING_OPERATORS:
        node_representation = parse_node(plan)
    else:
        node_representation = "STOP"
    node = TreeNode(node_representation)    
    last_child = None
    if "Plans" in plan:
        for sub_plan in plan["Plans"]:
            child_node = parse_plan(sub_plan)
            node.children.append(child_node)
            if last_child:
                last_child.sibling = child_node
            last_child = child_node
    return node
   

    
def parse_node(node):
# SeqScan_ / IndexOnlyScan_ / IndexScan_ ......
    node_representation = f"{node['Node Type'].replace(' ', '')}_"

    if node["Node Type"] == "Seq Scan":
        node_representation += f"{parse_seq_scan(node)}"
    elif node["Node Type"] == "Index Only Scan":
        node_representation += f"{parse_index_only_scan(node)}"
    elif node["Node Type"] == "Index Scan":
        node_representation += f"{parse_index_scan(node)}"
    elif node["Node Type"] == "CTE Scan":
        node_representation += f"{parse_cte_scan(node)}"
    elif node["Node Type"] == "Nested Loop":
        node_representation += f"{parse_nested_loop(node)}"
    elif node["Node Type"] == "Hash Join":
        node_representation += f"{parse_hash_join(node)}"
    elif node["Node Type"] == "Merge Join":
        node_representation += f"{parse_merge_join(node)}"
    elif node["Node Type"] == "Sort":
        node_representation += f"{parse_sort(node)}"
    else:  # : useless
        raise ValueError("_parse_node called with unsupported Node Type.")

    return node_representation

def stringify_attribute_columns(node, attribute):
    replacings = [(" ", ""), ("(", ""), (")", ""), ("[", ""), ("]", ""), ("::text", "")]
    remove_digits = str.maketrans("", "", digits)
    attribute_representation = f"{attribute.replace(' ', '')}_"
    if attribute not in node:
        return attribute_representation

    value = node[attribute]

    for replacee, replacement in replacings:
        value = value.replace(replacee, replacement)

    value = re.sub('".*?"', "", value)
    value = re.sub("'.*?'", "", value)
    value = value.translate(remove_digits)

    return value

def stringify_list_attribute(node, attribute):
    attribute_representation = f"{attribute.replace(' ', '')}_"
    if attribute not in node:
        return attribute_representation

    assert isinstance(node[attribute], list)
    value = node[attribute]

    for element in value:
        attribute_representation += f"{element}_"

    return attribute_representation

def parse_seq_scan(node):
    assert "Relation Name" in node

    node_representation = ""
    node_representation += f"{node['Relation Name']}_"

    node_representation += stringify_attribute_columns(node, "Filter")

    return node_representation

def parse_index_scan(node):
    assert "Relation Name" in node

    node_representation = ""
    node_representation += f"{node['Relation Name']}_"

    node_representation += stringify_attribute_columns(node, "Filter")
    node_representation += stringify_attribute_columns(node, "Index Cond")

    return node_representation

def parse_index_only_scan(node):
    assert "Relation Name" in node

    node_representation = ""
    node_representation += f"{node['Relation Name']}_"

    node_representation += stringify_attribute_columns(node, "Index Cond")

    return node_representation

def parse_cte_scan(node):
    assert "CTE Name" in node

    node_representation = ""
    node_representation += f"{node['CTE Name']}_"

    node_representation += stringify_attribute_columns(node, "Filter")

    return node_representation

def parse_nested_loop(node):
    node_representation = ""

    node_representation += stringify_attribute_columns(node, "Join Filter")

    return node_representation

def parse_hash_join(node):
    node_representation = ""

    node_representation += stringify_attribute_columns(node, "Join Filter")
    node_representation += stringify_attribute_columns(node, "Hash Cond")

    return node_representation

def parse_merge_join(node):
    node_representation = ""

    node_representation += stringify_attribute_columns(node, "Merge Cond")

    return node_representation

def parse_sort(node):
    node_representation = ""
    node_representation += stringify_list_attribute(node, "Sort Key")

    return node_representation