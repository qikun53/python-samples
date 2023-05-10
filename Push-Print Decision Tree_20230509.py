"""
Created on Sun May  7 14:54:27 2023

https://mljar.com/blog/visualize-decision-tree/
"""

#%% Example 6 - workable 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

import numpy as np


# Create a toy dataset with categorical variables
data = pd.DataFrame({
    'color': ['red', 'green', 'blue', 'red', 'green', 'blue', 'green', 'red', 'blue', 'blue', 'red', 'green'],
    'size': [1, 2,2,1,3,3, 2, 1,2, 3, 1,2],
    'shape': ['circle', 'circle', 'square', 'square', 'circle', 'square', 'circle', 'square', 'circle', 'square', 'circle', 'square'],
    'texture': ['smooth', 'smooth', 'rough', 'smooth', 'rough', 'smooth', 'smooth', 'rough', 'rough', 'smooth', 'smooth', 'rough'],
    'label': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
})

# Define the target encoder
encoder = TargetEncoder()

# Encode the categorical variables
X = data[['color', 'size', 'shape', 'texture']]
y = data['label']
X_encoded = encoder.fit_transform(X, y)

# Create a dictionary to map encoded values back to their original values
mapping_dict = {}
for col in X.columns:
    mapping_dict.update({f"{col}": dict(zip(X[col].unique(), X_encoded[col].unique()))})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree classifier with entropy criterion
clf = DecisionTreeClassifier(criterion='entropy')

# Train the decision tree on the encoded training data
clf.fit(X_train, y_train)

# Get the text representation of the tree
#text_representation = tree.export_text(clf, feature_names=X.columns.tolist(), show_weights=True)


#aux function, given a dictionary and a threshld, find the keys with values less than the threshold, and larger than the threshold
def findKeysLessThanThreshold (dictionary,threshold):
    keys_less_than_threshold = []
    for key, value in dictionary.items():
        if value <= threshold:
            keys_less_than_threshold.append(key)
    return keys_less_than_threshold
def findKeysLargerThanThreshold (dictionary,threshold):
    keys_larger_than_threshold = []
    for key, value in dictionary.items():
        if value > threshold:
            keys_larger_than_threshold.append(key)
    return keys_larger_than_threshold

from sklearn.tree import _tree
from sklearn.tree import export_text

def _compute_depth(tree, node):
    """
    Returns the depth of the subtree rooted in node.
    """

    def compute_depth_(
        current_node, current_depth, children_left, children_right, depths
    ):
        depths += [current_depth]
        left = children_left[current_node]
        right = children_right[current_node]
        if left != -1 and right != -1:
            compute_depth_(
                left, current_depth + 1, children_left, children_right, depths
            )
            compute_depth_(
                right, current_depth + 1, children_left, children_right, depths
            )

    depths = []
    compute_depth_(node, 1, tree.children_left, tree.children_right, depths)
    return max(depths)

def export_text_QW(
    decision_tree,
    *,
    threshold_optional=False,
    threshold_optional_lesser=[],
    threshold_optional_larger=[],
    feature_names=None,
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False,
):
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    feature_names : list of str, default=None
        A list of length n_features containing the feature names.
        If None generic names will be used ("feature_0", "feature_1", ...).

    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".

    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.

    decimals : int, default=2
        Number of decimal digits to display.

    show_weights : bool, default=False
        If true the classification weights will be exported on each leaf.
        The classification weights are the number of samples each class.

    Returns
    -------
    report : str
        Text summary of all the rules in the decision tree.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.tree import export_text
    >>> iris = load_iris()
    >>> X = iris['data']
    >>> y = iris['target']
    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    >>> decision_tree = decision_tree.fit(X, y)
    >>> r = export_text(decision_tree, feature_names=iris['feature_names'])
    >>> print(r)
    |--- petal width (cm) <= 0.80
    |   |--- class: 0
    |--- petal width (cm) >  0.80
    |   |--- petal width (cm) <= 1.75
    |   |   |--- class: 1
    |   |--- petal width (cm) >  1.75
    |   |   |--- class: 2
    """
    #check_is_fitted(decision_tree)
    tree_ = decision_tree.tree_
    #if is_classifier(decision_tree):
    #    class_names = decision_tree.classes_
    class_names = decision_tree.classes_
    right_child_fmt_num = "{} {} <= {}\n"
    right_child_fmt_cat = "{} {} in {}\n"
    left_child_fmt_num = "{} {} >  {}\n"
    left_child_fmt_cat = "{} {} in  {}\n"
    
    truncation_fmt = "{} {}\n"

    if max_depth < 0:
        raise ValueError("max_depth bust be >= 0, given %d" % max_depth)

    if feature_names is not None and len(feature_names) != tree_.n_features:
        raise ValueError(
            "feature_names must contain %d elements, got %d"
            % (tree_.n_features, len(feature_names))
        )
    if threshold_optional and ((threshold_optional_lesser==[])|(threshold_optional_larger==[])):
        raise ValueError('When turn on threshold_optional, both threshold_optional_lesser and threshold_optional_larger must not be blank list')
    
    
    if spacing <= 0:
        raise ValueError("spacing must be > 0, given %d" % spacing)

    if decimals < 0:
        raise ValueError("decimals must be >= 0, given %d" % decimals)

    if isinstance(decision_tree, DecisionTreeClassifier):
        value_fmt = "{}{} weights: {}\n"
        if not show_weights:
            value_fmt = "{}{}{}\n"
    else:
        value_fmt = "{}{} value: {}\n"

    if feature_names:
        feature_names_ = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else None
            for i in tree_.feature
        ]
    else:
        feature_names_ = ["feature_{}".format(i) for i in tree_.feature]

    export_text.report = ""

    def _add_leaf(value, class_name, indent):
        val = ""
        is_classification = isinstance(decision_tree, DecisionTreeClassifier)
        if show_weights or not is_classification:
            val = ["{1:.{0}f}, ".format(decimals, v) for v in value]
            val = "[" + "".join(val)[:-2] + "]"
        if is_classification:
            val += " class: " + str(class_name)
        export_text.report += value_fmt.format(indent, "", val)

    def print_tree_recurse(node, depth):
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)

        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]

        if depth <= max_depth + 1:
            info_fmt = ""
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names_[node]
                if threshold_optional: #Turn on the optional threshold to cope with the category variables
                    threshold_lesser=threshold_optional_lesser[node]
                    threshold_larger=threshold_optional_larger[node]
                    export_text.report += right_child_fmt_cat.format(indent, name, threshold_lesser)
                    export_text.report += info_fmt_left
                    print_tree_recurse(tree_.children_left[node], depth + 1)
                    
                    export_text.report += left_child_fmt_cat.format(indent, name, threshold_larger)
                    export_text.report += info_fmt_right
                    print_tree_recurse(tree_.children_right[node], depth + 1)
                else: #this is the default case
                    threshold = tree_.threshold[node]
                    threshold = "{1:.{0}f}".format(decimals, threshold)
                    export_text.report += right_child_fmt_num.format(indent, name, threshold)
                    export_text.report += info_fmt_left
                    print_tree_recurse(tree_.children_left[node], depth + 1)
    
                    export_text.report += left_child_fmt_num.format(indent, name, threshold)
                    export_text.report += info_fmt_right
                    print_tree_recurse(tree_.children_right[node], depth + 1)
            else:  # leaf
                _add_leaf(value, class_name, indent)
        else:
            subtree_depth = _compute_depth(tree_, node)
            if subtree_depth == 1:
                _add_leaf(value, class_name, indent)
            else:
                trunc_report = "truncated branch of depth %d" % subtree_depth
                export_text.report += truncation_fmt.format(indent, trunc_report)

    print_tree_recurse(0, 1)
    return export_text.report




# engineer the decision tree
feature_names=X_train.columns
left      = clf.tree_.children_left
right     = clf.tree_.children_right
threshold = clf.tree_.threshold
features  = [feature_names[i] for i in clf.tree_.feature]
threshold_indicator=[X.dtypes[feature] for feature in features] #indicates the types of the threshold as object, int 64 etc.
threshold_decoded_lesser=[]
threshold_decoded_larger=[]
for i in range(len(threshold)):
    if threshold_indicator[i]==object and threshold[i]!=-2:#if the threshold is decoded from object, threshold=-2 indicates a leaf
        threshold_decoded_lesser.append(findKeysLessThanThreshold(mapping_dict[features[i]],threshold[i]))
        threshold_decoded_larger.append(findKeysLargerThanThreshold(mapping_dict[features[i]],threshold[i]))
    else:
        threshold_decoded_lesser.append(threshold[i])
        threshold_decoded_larger.append(threshold[i])

text_representation_original = tree.export_text(clf, 
                                      feature_names=X.columns.tolist(), 
                                      show_weights=True)

text_representation2 = export_text_QW(clf, 
                                      threshold_optional=True,
                                      threshold_optional_lesser=threshold_decoded_lesser,
                                      threshold_optional_larger=threshold_decoded_larger,
                                      feature_names=X.columns.tolist(), 
                                      show_weights=True)


'''

# Replace encoded values with original values in the text representation
for col, mapping in mapping_dict.items():
    for key, value in mapping.items():
        text_representation = text_representation.replace(f"{col} <= {value:.4f}", f"{col} <= {key}")
'''
# Print the modified text representation
