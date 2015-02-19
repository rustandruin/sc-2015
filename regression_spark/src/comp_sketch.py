'''
This function computes a sketch for the given matrix.
parameters:
    matrix: a 'matrix' object
    objective: either 'x' or 'N'
    sketch_type: either 'projection' or 'sampling'
    projection_type: cs, gaussian, rademacher or srdht
'''

from utils import *
from rma_utils import *
from projections import *
import time
import os

def comp_sketch(matrix, objective, load_N=True, save_N=False, alg='cx', **kwargs):

    sketch_type = kwargs.get('sketch_type')

    if not os.path.exists('../N'):
        os.makedirs('../N')

    if objective == 'x':
        
        if sketch_type == 'projection':
            projection = Projections(**kwargs)
            t = time.time()
            x = projection.execute(matrix, 'x', alg, save_N)
            t = time.time() - t

            if save_N:
                N = [a[1][0] for a in x]
                x = [(a[0],a[1][1]) for a in x]
                #saving N
                filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k')))+ '.dat'
                data = {'N': N, 'time': t}
                pickle_write(filename,data)
 
        elif sketch_type == 'sampling':
            s = kwargs.get('s')
            new_N_proj = 0
            N_proj_filename = filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) +'.dat'

            if load_N and os.path.isfile(N_proj_filename):
                N_proj_filename = filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) +'.dat'
                result = pickle_load(N_proj_filename)
                N_proj = result['N']
                t_proj = result['time']
            else: #otherwise, compute it
                t = time.time()
                projection = Projections(**kwargs)
                N_proj = projection.execute(matrix, 'N', alg)
                t_proj = time.time() - t
                new_N_proj = 1

            sc = kwargs.pop('sc')
            t = time.time()
            sumLev = comp_lev_sum(matrix.matrix, sc, N_proj, alg) #computing leverage scores
            x = sample_solve(matrix.matrix, sc, N_proj, sumLev, s, save_N) #sampling and getting x
            t = time.time() - t + t_proj

            if save_N and new_N_proj:
                filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N_proj, 'time': t_proj}
                pickle_write(filename,data)

            if save_N:
                N = [a[1][0] for a in x]
                x = [(a[0],a[1][1]) for a in x]
                filename = '../N/N_' + matrix.name + '_sampling_s' + str(int(kwargs.get('s'))) + '_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N, 'time': t}
                pickle_write(filename,data)

        else:
            print 'Please enter a valid sketch type!'
        return x, t

    elif objective == 'N':
        if sketch_type == 'projection':
            N_proj_filename = filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'

            if load_N and os.path.isfile(N_proj_filename):
                result = pickle_load(filename)
                N = result['N']
                t = result['time']
            else:
                t = time.time()
                projection = Projections(**kwargs)
                N = projection.execute(matrix, 'N', alg)
                t = time.time() - t

                if save_N:
                    filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k')))+ '.dat'
                    data = {'N': N, 'time': t}
                    pickle_write(filename,data)

        elif sketch_type == 'sampling':
            s = kwargs.get('s')
            sc = kwargs.pop('sc')
            new_N_proj = 0
            new_N_samp = 0

            N_samp_filename = '../N/N_' + matrix.name + '_sampling_s' + str(int(kwargs.get('s'))) + '_' + kwargs.get('projection_type') + '_' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
            N_proj_filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'

            if load_N and os.path.isfile(N_samp_filename):
                result = pickle_load(filename)
                N = result['N']
                t = result['time']

            elif load_N and os.path.isfile(N_proj_filename):
                result = pickle_load(N_proj_filename)
                N_proj = result['N']
                t_proj = result['time']

            else:
                t = time.time()
                projection = Projections(**kwargs)
                N_proj = projection.execute(matrix, 'N', alg)
                t_proj = time.time() - t
                new_N_proj = 1

                t = time.time()
                sumLev = comp_lev_sum(matrix.matrix, sc, N_proj, alg)
                N = sample_svd(matrix.matrix, sc, N_proj, sumLev, s, alg)
                t = time.time() - t + t_proj
                new_N_samp = 1

            if save_N and new_N_proj:
                filename = '../N/N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N_proj, 'time': t_proj}
                pickle_write(filename,data)

            if save_N and new_N_samp:
                filename = '../N/N_' + matrix.name + '_sampling_s' + str(int(kwargs.get('s'))) + '_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N_proj, 'time': t}
                pickle_write(filename,data)

        else:
            raise ValueError('Please enter a valid sketch type!')
        return N, t
    else:
        raise ValueError('Please enter a valid objective!')


