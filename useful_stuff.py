import numpy as np

def find_edges_1D(arr):
  string_arr = ''.join(arr.astype(int).astype(str))
  edge_arr = string_arr.replace('01', '02').replace('10', '20') # Turn any True by a False to a 2
  edge_arr = (np.array(list(edge_arr)).astype(int) - 2).astype(bool)
  return ~edge_arr

def find_edges_2D(arr):
  x_arr = np.array([find_edges_1D(line) for line in arr])
  y_arr = np.swapaxes(np.array([find_edges_1D(line) for line in np.swapaxes(arr, 0, 1)]),0,1)
  edge_arr = x_arr | y_arr
  return edge_arr

def find_structure_edges(image, structure_id):
  image = ~(image - structure_id).astype(bool)
  image = find_edges_2D(image)
  return image

def find_structure_coords(image, structure_id):
  image = find_structure_edges(image, structure_id)
  y, x = np.where(image)
  return x, y

def find_structure_and_outline(image, structure_id):
  image = ~(image - structure_id).astype(bool)
  edges = find_edges_2D(image)
  y, x = np.where(edges)
  return image, x, y

def print_color(color):
  if type(color) is str: return f'\'{color}\''
  if type(color) is list: return '['+''.join([str(num)+',' for num in color])[:-2]+']'

def progress_word(current, total, word = 'lets just say, hehe, my peenitz'):
  length = len(word)
  progress = int(current / total * length // 1)
  print('\r|' + word[:progress] + ' '*int(length-progress), end = f'| {100*current/total:.2f}%')

def user_decision(options):
  print('Choose one of the options below')
  index_column_width = len(str(len(options)))
  for i, op in enumerate(options):
    index = str(i+1)
    index = ' '*(index_column_width-len(index)) + index
    print(f'{index} | {op}')

  ans = input()
  while(True):
    if ans in options: return ans
    elif ans.isdigit() and (int(ans) > 0) and (int(ans) <= len(options)): return options[int(ans)-1]
    else: ans = input(f'Input must be between 1 and {len(options)}')
