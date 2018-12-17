import numpy
import HW1_G1_Code2
import matplotlib.pyplot as plt 

"""
Mục tiêu tìm x thuộc [0; 5] để f(x)=4x**4 - 5x**3 + exp(-2x) - 7sin(x) -3cos(x) đạt giá trị lớn nhất
    
"""

chrm_per_pop = 20          #Số cá thể trong 1 quần thể
num_parents_mating = 10    #Số cặp lai ghép
num_bits=10                 #Số bit mã hóa 
mutation_rate= 0.01         #Tỉ lệ đột biến bằng 1%
epsilon=0.000005            #Độ lệch liên tiếp cho phép giữa các độ phù hợp liên tiếp
num_generations = 60 #số vòng lặp (thế hệ) sẽ thực hiện, có thể thay đổi
termination=0        #Cờ dừng vòng lặp
generation=0        
stall_generation=0
num_stall_generation=20     #Nếu có 20 giá trị độ phù hợp liên tiếp sai lệch nhau nhỏ hơn epsilon thì dừng vòng lặp
best_fit_indiv_generation=numpy.zeros((num_generations,1))    #Lưu cá thể tốt nhất qua từng thế hệ 

#Tạo quần thể mới
new_population=GA_extremal.init_pop(chrm_per_pop,num_bits)
print(new_population)

while (termination==0):
    generation=generation+1
    print("generation",generation)
        
    #Giải mã
    pop=GA_extremal.decode(new_population)
    
    #Tính toán độ phù hợp của cá thể trong quần thể
    fitness = GA_extremal.cal_pop_fitness(pop)

    #Lựa chọn các cá thể tốt nhất của quần thể
    
    best_fit = GA_extremal.select_mating_pool(new_population, fitness, num_parents_mating)
    
    #Cá thể tốt nhất của từng thế hệ
    best_fit_indiv_generation[generation-1,0]=GA_extremal.decode_ndarray(best_fit[0,:])
    print(GA_extremal.cal_pop_fitness(best_fit_indiv_generation[generation-1,0]))

    #Chọn ngẫu nhiên "chr_per_pop" cá thể từ nhóm cá thể tốt nhất
    rand_indx=GA_extremal.rand_indx_choosen(chrm_per_pop)

    parent= GA_extremal.indiv_from_rand_indx_choosen(rand_indx,best_fit)

    #Tiến hành lai ghép
    offspring=GA_extremal.crossover(parent,chrm_per_pop,num_bits)

    #Đột biến gen
    new_population=GA_extremal.mutation(offspring, mutation_rate)

    """
    Kiểm tra điều kiện dừng của thuật toán: 
    + Có 10 giá trị của độ phù hợp liên tiếp sai lệch nhỏ hơn epsilon
    + Hoặc trải qua 40 thế hệ 
    """

    if (generation==num_generations):
        termination=1
    elif (generation>1):
        if(numpy.absolute(GA_extremal.cal_pop_fitness(best_fit_indiv_generation[generation-1,0])-GA_extremal.cal_pop_fitness(best_fit_indiv_generation[generation-2,0]))<epsilon):
            stall_generation=stall_generation+1
            if(stall_generation==num_stall_generation):
                termination=1
        else:
            stall_generation=0


print("Best solution x = ", best_fit_indiv_generation[generation-1,0])
print("Maximum of function: ", GA_extremal.cal_pop_fitness(best_fit_indiv_generation[generation-1,0]))

plt.plot(GA_extremal.cal_pop_fitness(best_fit_indiv_generation[0:generation-1,0]))
plt.ylabel("Best fitness f(x)")
plt.xlabel("Generation")
plt.show()
