import numpy

def init_pop(num_individual, num_bits):
    pop=numpy.random.random_integers(low=0, high=1, size=(num_individual, num_bits))
    return pop

def decode(binary):
    #Giải mã mảng binary
    decimal=numpy.zeros((binary.shape[0],1))
    for i in range(binary.shape[0]):
        decimal[i]=0
        for j in range(binary.shape[1]):
            decimal[i]=decimal[i]+(binary[i,j])*2**j
        decimal[i]=+decimal[i]*5/(2**((binary.shape[1]))-1)
    return decimal

def decode_ndarray(binary):
    #Giải mã vector binary
    decimal=0
    for j in range(binary.shape[0]):
        decimal=decimal+(binary[j])*2**j
    decimal=decimal*5/(2**((binary.shape[0]))-1)
    return decimal

def cal_pop_fitness(pop):
    #Tính toán độ phù hợp
    fitness = 4*numpy.power(pop,4)-5*numpy.power(pop,3)+numpy.exp(-2*pop)-7*numpy.sin(pop)-3*numpy.cos(pop)   
    #fitness=numpy.power(pop,3)+3*numpy.power(pop,2)-4
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    #Lựa ra "num_parents" cá thể tốt nhất
    parents = numpy.zeros((num_parents, pop.shape[1]),dtype=numpy.uint8)   
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def rand_indx_choosen(N):
    #Lọc ngẫu nhiên để chọn ra N chỉ số 
    output=numpy.zeros((1,N),dtype=numpy.uint8)
    for i in range(N):
        x=numpy.random.uniform(low=0.0, high=1.0)
        count=numpy.histogram(x, [0.0, 0.21, 0.39, 0.54, 0.67, 0.77, 0.84, 0.91, 0.94, 0.97, 1.0])  #mảng có "num_parents_mating" phần tử
        count=count[0]
        indx=numpy.where(count==1)
        indx=indx[0][0]
        output[0,i]=indx   
    return output   

def indiv_from_rand_indx_choosen(rand_indx, parent):
    #Chọn ra N cá thể từ parent dựa vào chỉ số được chọn ngẫu nhiên rand_indx
    pop_choosen=numpy.zeros((rand_indx.shape[1],parent.shape[1]),dtype=numpy.uint8)
    indx=rand_indx[0,:]   
    for i in range(rand_indx.shape[1]): 
        pop_choosen[i,:]=parent[indx[i],:]
    return pop_choosen


def crossover(parents, pop_size, n_bits):
    offspring = numpy.zeros((pop_size,n_bits),dtype=numpy.uint8)
    # Lấy điểm giao crossover
    crossover_point = numpy.uint8(n_bits/2)
    for k in range(numpy.uint8(pop_size/2)):
        #Chỉ sô của parents đầu tiên
        parent1_idx = k
        #Chỉ số của parents thứ 2
        parent2_idx = k+numpy.uint8(pop_size/2)
        # lấy 1 nửa gens của parents 1
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # lấy 1 nửa gens của parents 2
        offspring[k, crossover_point:n_bits+1] = parents[parent2_idx, crossover_point:n_bits+1]
         # lấy 1 nửa gens của parents 2
        offspring[k+numpy.uint8(pop_size/2), 0:crossover_point] = parents[parent2_idx, 0:crossover_point]
        # lấy 1 nửa gens của parents 1
        offspring[k+numpy.uint8(pop_size/2), crossover_point:n_bits+1] = parents[parent1_idx, crossover_point:n_bits+1]
    return offspring

def mutation(offspring_crossover, rate):
    #đột biến một gen bất kì trong quần thể
    n_mutation=numpy.uint8(numpy.round(rate*offspring_crossover.shape[0]*offspring_crossover.shape[1]))
    for i in range (n_mutation):
        rand_chro=numpy.random.random_integers(low=0, high=offspring_crossover.shape[0]-1)
        rand_gene=numpy.random.random_integers(low=0, high=offspring_crossover.shape[1]-1)
        offspring_crossover[rand_chro,rand_gene]=numpy.uint8(1-offspring_crossover[rand_chro,rand_gene])
    return offspring_crossover