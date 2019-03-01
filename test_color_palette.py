import color_palette

def test_color_difference():
    col1 = [200,0,0]
    col2 = [200,3,4]
    result = 5
    assert color_palette.color_difference(col1, col2) == result
