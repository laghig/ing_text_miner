import string

def first_five_ing(text):
    """
    Input: string of the ingredients text separated by comma
    Output: string with only the fist five ingredients
    """
    text = ",".join(text.split(",",1)[:-1])
    return text


if __name__ == "__main__":
    print('hello')