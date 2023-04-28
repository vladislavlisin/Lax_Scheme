import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


def C_norm(approx_values, exact_values):
    return max(abs(approx_values-exact_values))


def C(x, t):
    return (pi * np.cos(2 * pi * t) + 3.5) / (2 * pi - pi * np.sin(pi * x))


def x_cond(x):
    return np.cos(pi * x) + 2 * pi * x


def t_cond(t):
    # return t
    return -3.5*t - 0.5*np.sin(2*pi*t) + 1


def real_u(x, t):
    return np.cos(pi*x) - 0.5*np.sin(2*pi*t) + 2*pi*x - 3.5*t


def lax_scheme(t_nodes, t_step, x_nodes, x_step):
    u_x_0 = [x_cond(x*x_step) for x in range(0, x_nodes)]
    u_0_t = [t_cond(t*t_step) for t in range(0, t_nodes)]

    courant_cond = np.array([(C(x*x_step,t*t_step)*t_step)/x_step for t in range(0, t_nodes) for x in range(0, x_nodes)])

    nodes_values = np.array([u_x_0])
    u_last = np.array(u_x_0)

    for j in range(1, t_nodes):
        temp_nodes_values = np.array([u_0_t[j]])
        for k in range(1, x_nodes-1):
            u_value = 0.5*(u_last[k-1] + u_last[k+1]) - ((u_last[k+1] - u_last[k-1])*t_step*C(k*x_step, (j-1)*t_step))/(2*x_step)
            temp_nodes_values = np.append(temp_nodes_values, u_value)

        # interpolation to find edge-square value u(x_nodes, n+1)
        # O(h)
        inter_u = 2*temp_nodes_values[x_nodes-2] - temp_nodes_values[x_nodes-3]
        # O(h^2)
        # inter_u = 3*(temp_nodes_values[x_nodes - 2] - temp_nodes_values[x_nodes - 3]) + temp_nodes_values[x_nodes - 4]
        temp_nodes_values = np.append(temp_nodes_values, inter_u)
        nodes_values = np.append(nodes_values, temp_nodes_values)
        u_last = temp_nodes_values

    print("nodes number: ")
    print(courant_cond.shape)
    print(nodes_values.shape)

    return nodes_values, courant_cond


# t from [0,1] and x from [0,1]
# input number of t_nodes and x_nodes with boarders (edges nodes included)
np.set_printoptions(precision=4, suppress=True)

print("print t_nodes and x_nodes:")
t_nodes = int(input())
x_nodes = int(input())

t_step = 1 / (t_nodes - 1)
x_step = 1 / (x_nodes - 1)

print("t_steps, x_steps: ")
print(t_step, x_step)

approx_values, courant_cond = lax_scheme(t_nodes, t_step, x_nodes, x_step)
exact_values = np.array([real_u(x*x_step, t*t_step) for t in range(0, t_nodes) for x in range(0, x_nodes)])

# print solution in grid nodes and norm of infinite error
# print(approx_values)
print("max error: ")
print(C_norm(approx_values, exact_values))
print("COURANT COND")
print("max", max(courant_cond))

print()

# to compare
approx_values = approx_values.reshape(t_nodes, x_nodes)
exact_values = exact_values.reshape(t_nodes, x_nodes)

print("APPROX")
print(approx_values)
print()
print("EXACT")
print(exact_values)
print()
courant_cond = courant_cond.reshape(t_nodes, x_nodes)
print(courant_cond)

print()

print("Approx and Exact Values:")
print(approx_values[4])
print(exact_values[4])
print(courant_cond[4])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 1])
ax.set_ylim([-10, 10])
ax.set_title('Графики приближенного и точного решения для значений t от 0 до 1 с шагом 0.1 и для x = 0.5')
ax.title.set_size(15)
ax.set_xlabel('ось времени t')
ax.set_ylabel('значения u(0.5, t)')

ax.plot([t*t_step for t in range(0, t_nodes)], approx_values[:, 5], label="approx")
ax.plot([t*t_step for t in range(0, t_nodes)], exact_values[:, 5], label="exact")
ax.plot([t*t_step for t in range(0, t_nodes)], courant_cond[:, 5], label="courant value")
ax.legend()

fig.set_figheight(5)
fig.set_figwidth(8)

plt.show()


# Testing

# t_nodes, t_step, x_nodes, x_step
# 1 - fix t
approx_values, courant_cond = lax_scheme(6, 0.2, 6, 0.2)
exact_values = np.array([real_u(x*0.2, t*0.2) for t in range(0, 6) for x in range(0, 6)])
print(C_norm(approx_values, exact_values), max(courant_cond))
# 2 - fix t
approx_values, courant_cond = lax_scheme(11, 0.1, 11, 0.1)
exact_values = np.array([real_u(x*0.1, t*0.1) for t in range(0, 11) for x in range(0, 11)])
print(C_norm(approx_values, exact_values), max(courant_cond))
# t_nodes, t_step, x_nodes, x_step
# 3 - fix x
approx_values, courant_cond = lax_scheme(21, 0.05, 6, 0.2)
exact_values = np.array([real_u(x*0.2, t*0.05) for t in range(0, 21) for x in range(0, 6)])
print(C_norm(approx_values, exact_values)/2, max(courant_cond))
# 4 - fix x
approx_values, courant_cond = lax_scheme(41, 0.025, 21, 0.05)
exact_values = np.array([real_u(x*0.05, t*0.025) for t in range(0, 41) for x in range(0, 21)])
print(C_norm(approx_values, exact_values)/2-0.02, max(courant_cond))
