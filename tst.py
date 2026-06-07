def find_overlap(s1, s2):
    # Check for overlap at the end of s1 and beginning of s2
    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i:] == s2[:i]:
            return i
    # Check for overlap at the end of s2 and beginning of s1
    for i in range(1, min(len(s1), len(s2)) + 1):
        if s2[-i:] == s1[:i]:
            return -i
    return 0


def combine_strings_with_overlap(strings):
    combined = []
    while strings:
        current_string = strings.pop(0)
        found_overlap = False
        for i, other_string in enumerate(strings):
            overlap_length = find_overlap(current_string, other_string)
            if overlap_length != 0:
                # Combine the strings based on the overlap length
                if overlap_length > 0:
                    combined.append(current_string + other_string[overlap_length:])
                else:
                    combined.append(other_string + current_string[-overlap_length:])
                found_overlap = True
                del strings[i]
                break
        if not found_overlap:
            combined.append(current_string)
    return combined


# Example usage:
strings = ["hello world", "world is nigh!", "bums", "nigh! on impossible"]
result = combine_strings_with_overlap(strings)
print(result)  # Output might be ['abcdefg', 'xyza']
