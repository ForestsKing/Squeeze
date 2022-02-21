from exp.exp import Exp

if __name__ == '__main__':
    exp = Exp(input_path='data/B0/B_cuboid_layer_2_n_ele_2/',
              output_path='output/',
              columns=['a', 'b', 'c', 'd'],
              derived=False)
    exp.location()
    exp.evaluation()
