import json

class TreeNode:
    def __init__(self, node_type, children=None):
        self.node_type = node_type
        self.children = children if children is not None else []

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.node_type) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

def build_tree(plan):
    node_type = plan.get("Node Type", "Unknown")
    children = plan.get("Plans", [])
    tree_node = TreeNode(node_type)
    for child in children:
        tree_node.children.append(build_tree(child))
    return tree_node

def json_to_tree(json_data):
    plans = json.loads(json_data)
    root_plan = plans[0]["Plan"]
    return build_tree(root_plan)

def find_shared_parents(tree1, tree2):
    shared_parents = []

    def traverse(t1, t2):
        if t1.node_type == t2.node_type:
            shared_parents.append(t1.node_type)
            for c1, c2 in zip(t1.children, t2.children):
                traverse(c1, c2)

    traverse(tree1, tree2)
    return shared_parents
# Example usage
json_data_indexed = '''
[
  {
    "Plan": {
      "Node Type": "Limit",
      "Parallel Aware": false,
      "Startup Cost": 1007869.52,
      "Total Cost": 1007869.53,
      "Plan Rows": 1,
      "Plan Width": 32,
      "Actual Startup Time": 13575.538,
      "Actual Total Time": 13878.481,
      "Actual Rows": 1,
      "Actual Loops": 1,
      "Plans": [
        {
          "Node Type": "Aggregate",
          "Strategy": "Plain",
          "Partial Mode": "Finalize",
          "Parent Relationship": "Outer",
          "Parallel Aware": false,
          "Startup Cost": 1007869.52,
          "Total Cost": 1007869.53,
          "Plan Rows": 1,
          "Plan Width": 32,
          "Actual Startup Time": 13379.879,
          "Actual Total Time": 13682.820,
          "Actual Rows": 1,
          "Actual Loops": 1,
          "Plans": [
            {
              "Node Type": "Gather",
              "Parent Relationship": "Outer",
              "Parallel Aware": false,
              "Startup Cost": 1007869.46,
              "Total Cost": 1007869.47,
              "Plan Rows": 4,
              "Plan Width": 64,
              "Actual Startup Time": 13378.595,
              "Actual Total Time": 13682.802,
              "Actual Rows": 5,
              "Actual Loops": 1,
              "Workers Planned": 4,
              "Workers Launched": 4,
              "Single Copy": false,
              "Plans": [
                {
                  "Node Type": "Aggregate",
                  "Strategy": "Plain",
                  "Partial Mode": "Partial",
                  "Parent Relationship": "Outer",
                  "Parallel Aware": false,
                  "Startup Cost": 1007859.46,
                  "Total Cost": 1007859.47,
                  "Plan Rows": 1,
                  "Plan Width": 64,
                  "Actual Startup Time": 13359.863,
                  "Actual Total Time": 13359.864,
                  "Actual Rows": 1,
                  "Actual Loops": 5,
                  "Plans": [
                    {
                      "Node Type": "Nested Loop",
                      "Parent Relationship": "Outer",
                      "Parallel Aware": false,
                      "Join Type": "Inner",
                      "Startup Cost": 0.56,
                      "Total Cost": 1004523.20,
                      "Plan Rows": 190643,
                      "Plan Width": 33,
                      "Actual Startup Time": 131.049,
                      "Actual Total Time": 13291.042,
                      "Actual Rows": 154198,
                      "Actual Loops": 5,
                      "Inner Unique": false,
                      "Plans": [
                        {
                          "Node Type": "Seq Scan",
                          "Parent Relationship": "Outer",
                          "Parallel Aware": true,
                          "Relation Name": "part",
                          "Alias": "part",
                          "Startup Cost": 0.00,
                          "Total Cost": 45962.00,
                          "Plan Rows": 500000,
                          "Plan Width": 25,
                          "Actual Startup Time": 0.017,
                          "Actual Total Time": 65.423,
                          "Actual Rows": 400000,
                          "Actual Loops": 5
                        },
                        {
                          "Node Type": "Index Scan",
                          "Parent Relationship": "Inner",
                          "Parallel Aware": false,
                          "Scan Direction": "Forward",
                          "Index Name": "idx_partkey",
                          "Relation Name": "lineitem",
                          "Alias": "lineitem",
                          "Startup Cost": 0.56,
                          "Total Cost": 1.91,
                          "Plan Rows": 1,
                          "Plan Width": 16,
                          "Actual Startup Time": 0.027,
                          "Actual Total Time": 0.032,
                          "Actual Rows": 0,
                          "Actual Loops": 2000000,
                          "Index Cond": "(l_partkey = part.p_partkey)",
                          "Rows Removed by Index Recheck": 0,
                          "Filter": "((l_shipdate >= '1994-10-01'::date) AND (l_shipdate < '1994-11-01 00:00:00'::timestamp without time zone))",
                          "Rows Removed by Filter": 30
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    },
    "Planning Time": 2.194,
    "Triggers": [],
    "JIT": {
      "Worker Number": -1,
      "Functions": 53,
      "Options": {
        "Inlining": true,
        "Optimization": true,
        "Expressions": true,
        "Deforming": true
      },
      "Timing": {
        "Generation": 7.537,
        "Inlining": 247.223,
        "Optimization": 373.934,
        "Emission": 222.525,
        "Total": 851.220
      }
    },
    "Execution Time": 13921.487
  }
]
'''
json_data_noindex = '''
[
  {
    "Plan": {
      "Node Type": "Limit",
      "Parallel Aware": false,
      "Startup Cost": 1307775.94,
      "Total Cost": 1307775.96,
      "Plan Rows": 1,
      "Plan Width": 32,
      "Actual Startup Time": 2226.279,
      "Actual Total Time": 2415.862,
      "Actual Rows": 1,
      "Actual Loops": 1,
      "Plans": [
        {
          "Node Type": "Aggregate",
          "Strategy": "Plain",
          "Partial Mode": "Finalize",
          "Parent Relationship": "Outer",
          "Parallel Aware": false,
          "Startup Cost": 1307775.94,
          "Total Cost": 1307775.96,
          "Plan Rows": 1,
          "Plan Width": 32,
          "Actual Startup Time": 2040.325,
          "Actual Total Time": 2229.907,
          "Actual Rows": 1,
          "Actual Loops": 1,
          "Plans": [
            {
              "Node Type": "Gather",
              "Parent Relationship": "Outer",
              "Parallel Aware": false,
              "Startup Cost": 1307775.89,
              "Total Cost": 1307775.90,
              "Plan Rows": 4,
              "Plan Width": 64,
              "Actual Startup Time": 2037.723,
              "Actual Total Time": 2229.889,
              "Actual Rows": 5,
              "Actual Loops": 1,
              "Workers Planned": 4,
              "Workers Launched": 4,
              "Single Copy": false,
              "Plans": [
                {
                  "Node Type": "Aggregate",
                  "Strategy": "Plain",
                  "Partial Mode": "Partial",
                  "Parent Relationship": "Outer",
                  "Parallel Aware": false,
                  "Startup Cost": 1307765.89,
                  "Total Cost": 1307765.90,
                  "Plan Rows": 1,
                  "Plan Width": 64,
                  "Actual Startup Time": 2004.389,
                  "Actual Total Time": 2004.391,
                  "Actual Rows": 1,
                  "Actual Loops": 5,
                  "Plans": [
                    {
                      "Node Type": "Hash Join",
                      "Parent Relationship": "Outer",
                      "Parallel Aware": true,
                      "Join Type": "Inner",
                      "Startup Cost": 1254445.28,
                      "Total Cost": 1304429.63,
                      "Plan Rows": 190643,
                      "Plan Width": 33,
                      "Actual Startup Time": 1771.912,
                      "Actual Total Time": 1952.288,
                      "Actual Rows": 154198,
                      "Actual Loops": 5,
                      "Inner Unique": false,
                      "Hash Cond": "(part.p_partkey = lineitem.l_partkey)",
                      "Plans": [
                        {
                          "Node Type": "Seq Scan",
                          "Parent Relationship": "Outer",
                          "Parallel Aware": true,
                          "Relation Name": "part",
                          "Alias": "part",
                          "Startup Cost": 0.00,
                          "Total Cost": 45962.00,
                          "Plan Rows": 500000,
                          "Plan Width": 25,
                          "Actual Startup Time": 0.007,
                          "Actual Total Time": 52.244,
                          "Actual Rows": 400000,
                          "Actual Loops": 5
                        },
                        {
                          "Node Type": "Hash",
                          "Parent Relationship": "Inner",
                          "Parallel Aware": true,
                          "Startup Cost": 1253083.54,
                          "Total Cost": 1253083.54,
                          "Plan Rows": 108939,
                          "Plan Width": 16,
                          "Actual Startup Time": 1771.042,
                          "Actual Total Time": 1771.043,
                          "Actual Rows": 154198,
                          "Actual Loops": 5,
                          "Hash Buckets": 1048576,
                          "Original Hash Buckets": 1048576,
                          "Hash Batches": 1,
                          "Original Hash Batches": 1,
                          "Peak Memory Usage": 49056,
                          "Plans": [
                            {
                              "Node Type": "Seq Scan",
                              "Parent Relationship": "Outer",
                              "Parallel Aware": true,
                              "Relation Name": "lineitem",
                              "Alias": "lineitem",
                              "Startup Cost": 0.00,
                              "Total Cost": 1253083.54,
                              "Plan Rows": 108939,
                              "Plan Width": 16,
                              "Actual Startup Time": 161.245,
                              "Actual Total Time": 1724.058,
                              "Actual Rows": 154198,
                              "Actual Loops": 5,
                              "Filter": "((l_shipdate >= '1994-10-01'::date) AND (l_shipdate < '1994-11-01 00:00:00'::timestamp without time zone))",
                              "Rows Removed by Filter": 11843012
                            }
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    },
    "Planning Time": 0.649,
    "Triggers": [
    ],
    "JIT": {
      "Worker Number": -1,
      "Functions": 78,
      "Options": {
        "Inlining": true,
        "Optimization": true,
        "Expressions": true,
        "Deforming": true
      },
      "Timing": {
        "Generation": 16.993,
        "Inlining": 203.464,
        "Optimization": 502.555,
        "Emission": 284.831,
        "Total": 1007.843
      }
    },
    "Execution Time": 2420.244
  }
]
'''
tree_indexed = json_to_tree(json_data_indexed)
tree_noindex = json_to_tree(json_data_noindex)

shared_parents = find_shared_parents(tree_indexed, tree_noindex)
print("Shared parents:",shared_parents)
