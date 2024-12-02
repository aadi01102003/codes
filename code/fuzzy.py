def fuzzy_union(set1, set2):
    """
    Computes the union of two fuzzy sets.
    
    Union: max(a, b)
    """
    return [max(a, b) for a, b in zip(set1, set2)]

def fuzzy_intersection(set1, set2):
    """
    Computes the intersection of two fuzzy sets.
    
    Intersection: min(a, b)
    """
    return [min(a, b) for a, b in zip(set1, set2)]

def fuzzy_complement(fuzzy_set):
    """
    Computes the complement of a fuzzy set.
    
    Complement: 1 - a
    """
    return [1 - a for a in fuzzy_set]

# Example fuzzy sets
set1 = [0.1, 0.4, 0.7, 0.8]
set2 = [0.5, 0.3, 0.6, 0.9]

# Fuzzy operations
union_result = fuzzy_union(set1, set2)
intersection_result = fuzzy_intersection(set1, set2)
complement_set1 = fuzzy_complement(set1)
complement_set2 = fuzzy_complement(set2)

# Display results
print("Set 1:", set1)
print("Set 2:", set2)
print("Union:", union_result)
print("Intersection:", intersection_result)
print("Complement of Set 1:", complement_set1)
print("Complement of Set 2:", complement_set2)
