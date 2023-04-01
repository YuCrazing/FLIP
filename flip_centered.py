from operator import is_
from typing import List
import taichi as ti
import numpy as np
import time


# apply force after grid normalization (or explosion)
# pressure in air cells should be 0 (or volume shrink quickly)
# velocity in air cells may not be 0

# TODO: velocities which are next to solid should be extraploated before and after solving pressure 


ti.init(arch=ti.cuda, default_fp=ti.f32, debug=False)


res = 512 * 3
dt = 1.6e-2 #2e-3 #2e-2


rho = 1000
jacobi_iters = 500
jacobi_damped_para = 0.67
FLIP_blending = 0.0



m_g = 128
n_grid = m_g*m_g
n_particle = n_grid*4

length = 1.0
dx = length/m_g
inv_dx = 1/dx

# solid boundary
boundary_width = 2

eps = 1e-5

# show grid types
debug = False


# MAC grid
velocities = ti.Vector.field(2, dtype=ti.f32, shape=(m_g, m_g))
velocities_before_projection = ti.Vector.field(2, dtype=ti.f32, shape=(m_g, m_g))

last_velocities = ti.Vector.field(2, dtype=ti.f32, shape=(m_g, m_g))

weights = ti.field(dtype=ti.f32, shape=(m_g, m_g))


pressures = ti.field(dtype=ti.f32, shape=(m_g, m_g))
new_pressures = ti.field(dtype=ti.f32, shape=(m_g, m_g))
divergences = ti.field(dtype=ti.f32, shape=(m_g, m_g))

FLUID = 0
AIR = 1
SOLID = 2

types = ti.field(dtype=ti.i32, shape=(m_g, m_g))

particle_velocity = ti.Vector.field(2, dtype=ti.f32, shape=n_particle)
particle_position = ti.Vector.field(2, dtype=ti.f32, shape=n_particle)



@ti.kernel
def init_grid():
    for i, j in types:
        if i < boundary_width or i >= m_g-boundary_width or j < boundary_width or j >= m_g-boundary_width:
            types[i, j] = SOLID


# should not generate particles in solid cells
@ti.kernel
def init_particle():
    for i in particle_position:
        particle_position[i] = (ti.Vector([ti.random(), ti.random()])*0.5 + ti.Vector([0.2, 0.15])) * length
        particle_velocity[i] = ti.Vector([0.0, 0.0])


@ti.func
def is_valid(i, j):
    return i >= 0 and i <= m_g-1 and j >= 0 and j <= m_g-1

@ti.func
def is_solid(i, j):
    return (not is_valid(i, j)) or types[i, j] == SOLID

@ti.func
def is_air(i, j):
    return is_valid(i, j) and types[i, j] == AIR

@ti.func
def is_fluid(i, j):
    return is_valid(i, j) and types[i, j] == FLUID


@ti.kernel
def handle_boundary():
    for i, j in velocities:
        if is_solid(i, j):
            velocities[i, j] = [0.0, 0.0]

@ti.kernel
def init_step():

    for i, j in types:
        if not is_solid(i, j):
            types[i, j] = AIR

    for k in particle_velocity:
        grid = (particle_position[k] * inv_dx).cast(int)
        if not is_solid(grid.x, grid.y):
            types[grid] = FLUID


    for k in ti.grouped(velocities):
        velocities[k] = [0.0, 0.0]
        weights[k] = 0.0
  
    for k in ti.grouped(pressures):
        if is_air(k.x, k.y):
            pressures[k] = 0.0
            new_pressures[k] = 0.0

neighbors = ti.Vector.field(2, dtype=ti.i32, shape=100)
neighbor_count = ti.field(dtype=ti.i32, shape=1)
pid_to_check = 826

@ti.func
def scatter(grid_v, grid_m, xp, vp, stagger, pid):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    # if pid == pid_to_check:
    #     neighbor_count[0] = 0
    #     print(xp, base, (xp*inv_dx).cast(ti.i32))

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            # dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * vp
            grid_m[base + offset] += weight
            # if pid == pid_to_check:
            #     neighbors[neighbor_count[0]] = (base + offset)
            #     neighbor_count[0] += 1


@ti.kernel
def particle_to_grid():

    for k in particle_position:

        pos = particle_position[k]
        vel = particle_velocity[k]

        # stagger_u = ti.Vector([0.0, 0.5])
        # stagger_v = ti.Vector([0.5, 0.0])
        stagger_u = ti.Vector([0.5, 0.5])

        scatter(velocities, weights, pos, vel, stagger_u, k)


@ti.kernel
def grid_normalization():

    for k in ti.grouped(velocities):
        weight = weights[k]
        if weight > 0:
            velocities[k] = velocities[k] / weight



@ti.kernel
def apply_gravity():
    for i, j in velocities:
        # if not is_solid(i, j-1) and not is_solid(i, j):
            velocities[i, j].y += -9.8 * dt



@ti.func
def sample(i, j, field: ti.template()):
    if i < boundary_width:
        i = boundary_width
    if i >= m_g - boundary_width:
        i = m_g - boundary_width-1
    if j < boundary_width:
        j = boundary_width
    if j >= m_g - boundary_width:
        j = m_g - boundary_width-1
    return field[i, j]

@ti.kernel
def solve_divergence():

    for i, j in divergences:
        # if is_fluid(i, j):
        if weights[i, j] > 0 and not is_solid(i, j):

            v_l = velocities[i-1, j].x
            v_r = velocities[i+1, j].x
            v_d = velocities[i, j-1].y
            v_u = velocities[i, j+1].y

            factor = 1.0
            if is_solid(i-1, j): 
                v_l = -velocities[i, j].x * factor
            if is_solid(i+1, j): 
                v_r = -velocities[i, j].x * factor
            if is_solid(i, j-1): 
                v_d = -velocities[i, j].y * factor
            if is_solid(i, j+1): 
                v_u = -velocities[i, j].y * factor

            div = v_r - v_l + v_u - v_d

            divergences[i, j] = div / (2*dx)


@ti.kernel
def pressure_jacobi(p:ti.template(), new_p:ti.template()):

    w = jacobi_damped_para

    for i, j in p:
        # if is_fluid(i, j):
        if weights[i, j] > 0 and not is_solid(i, j):

            p_l = 0.0
            p_r = 0.0
            p_d = 0.0
            p_u = 0.0


            k = 4
            if is_solid(i-1, j):
                p_l = 0.0
                k -= 1
            else:
                p_l = p[i-1, j]
            
            if is_solid(i+1, j):
                p_r = 0.0
                k -= 1
            else:
                p_r = p[i+1, j]

            if is_solid(i, j-1):
                p_d = 0.0
                k -= 1
            else:
                p_d = p[i, j-1]

            if is_solid(i, j+1):
                p_u = 0.0
                k -= 1
            else:
                p_u = p[i, j+1]


            new_p[i, j] = (1 - w) * p[i, j] + w * ( p_l + p_r + p_d + p_u - divergences[i, j] * rho / dt * (dx*dx) ) / k



@ti.kernel
def projection():
    # if debug:
    #     print("v0")
    #     ti.loop_config(serialize=True)
    #     for j in range(m_g):
    #         for i in range(m_g):
    #             print(velocities[i, j].y, end=" ")
    #         print("")
    for i, j in ti.ndrange(m_g, m_g):
        velocities_before_projection[i, j] = velocities[i, j]
        # if is_fluid(i, j):
        if weights[i, j] > 0 and not is_solid(i, j): 
        # if not is_solid(i, j):
            # if is_solid(i-1, j) or is_solid(i+1, j):
            #     velocities_u[i, j] = 0.0
            # else:
                grad_p = ti.Vector([pressures[i+1, j]-pressures[i, j], pressures[i, j+1]-pressures[i, j]]) / dx
                # if is_solid(i-1, j):
                #     grad_p.x = (pressures[i+1, j]-pressures[i, j])/dx
                if is_solid(i+1, j):
                    grad_p.x = (pressures[i, j]-pressures[i-1, j])/dx
                # if is_solid(i, j-1):
                    # grad_p.y = (pressures[i, j+1]-pressures[i, j])/dx
                if is_solid(i, j+1):
                    grad_p.y = (pressures[i, j]-pressures[i, j-1])/dx
                velocities[i, j] -= grad_p / rho * dt
                # velocities[i, j] -= ti.Vector([sample(i+1, j, pressures) - sample(i-1, j, pressures), sample(i, j+1, pressures) - sample(i, j-1, pressures)]) / (dx*2) / rho * dt
                # if is_solid(i-1, j) and velocities[i, j].x < 0:
                #     velocities[i, j].x = 0
                #     velocities[i-1, j].x = 0
                # if is_solid(i+1, j) and velocities[i, j].x > 0:
                #     velocities[i, j].x = 0
                #     velocities[i+1, j].x = 0
                # if is_solid(i, j-1) and velocities[i, j].y < 0:
                #     velocities[i, j].y = 0
                #     velocities[i, j-1].y = 0
                # if is_solid(i, j+1) and velocities[i, j].y > 0:
                #     velocities[i, j].y = 0
                #     velocities[i, j+1].y = 0
                # if debug and is_solid(i+1, j):
                #     print(i, j, sample(i+1, j, pressures), sample(i-1, j, pressures))
        else: # TODO
            if is_solid(i, j) and weights[i, j] > 0:
                # velocities[i, j] = [0.0, 0.0]
                if is_solid(i-1, j) and velocities[i, j].x < 0:
                    velocities[i, j].x = 0
                if is_solid(i+1, j) and velocities[i, j].x > 0:
                    velocities[i, j].x = 0
                if is_solid(i, j-1) and velocities[i, j].y < 0:
                    velocities[i, j].y = 0
                if is_solid(i, j+1) and velocities[i, j].y > 0:
                    velocities[i, j].y = 0
    # if debug:    
    #     print("v")
    #     ti.loop_config(serialize=True)
    #     for j in range(m_g):
    #         for i in range(m_g):
    #             print(velocities[i, m_g-1-j].y, end=" ")
    #         print("")

    #     print("dv")
    #     ti.loop_config(serialize=True)
    #     for j in range(m_g):
    #         for i in range(m_g):
    #             print((sample(i, m_g-1-j+1, pressures) - sample(i, m_g-1-j-1, pressures)) / (dx*2) / rho * dt, end=" ")
    #         print("")

    #     print("p")
    #     ti.loop_config(serialize=True)
    #     for j in range(m_g):
    #         for i in range(m_g):
    #             print(pressures[i, m_g-1-j], end=" ")
    #         print("")
    
@ti.func
def gather(grid_v, last_grid_v, xp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    # base = (xp * inv_dx).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
    # fx = xp * inv_dx - (base.cast(ti.f32))

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    v_pic = ti.Vector([0.0, 0.0])
    v_flip = ti.Vector([0.0, 0.0])

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            v_pic  += weight * grid_v[base + offset]
            v_flip += weight * (grid_v[base + offset] - last_grid_v[base + offset])

    return v_pic, v_flip


@ti.kernel
def grid_to_particle():

    # stagger_u = ti.Vector([0.0, 0.5])
    # stagger_v = ti.Vector([0.5, 0.0])
    stagger_u = ti.Vector([0.5, 0.5])
    
    for k in ti.grouped(particle_position):
    
        p = particle_position[k]

        pic_x, flip_dx = gather(velocities, last_velocities, p, stagger_u)

        pic_vel = pic_x
        flip_vel = particle_velocity[k] + flip_dx

        particle_velocity[k] = (1.0-FLIP_blending) * pic_vel + FLIP_blending * flip_vel


@ti.kernel
def advect_particles():

    for k in ti.grouped(particle_position):

        pos = particle_position[k]
        vel = particle_velocity[k]
        
        pos += vel * dt

        if pos.x < dx * boundary_width:
            pos.x = dx * boundary_width + eps
            vel.x = 0
        if pos.x >= length - dx * boundary_width:
            pos.x = length - dx * boundary_width - eps
            vel.x = 0

        if pos.y < dx * boundary_width:
            pos.y = dx * boundary_width + eps
            vel.y = 0
        if pos.y >= length - dx * boundary_width:
            pos.y = length - dx * boundary_width - eps
            vel.y = 0


        particle_position[k] = pos
        particle_velocity[k] = vel

@ti.kernel
def fill_matrix(A: ti.types.sparse_matrix_builder(), F_b: ti.types.ndarray()):
    
    for I in ti.grouped(divergences):
        F_b[I[0] * m_g + I[1]] = - divergences[I] * dx * dx * rho / dt
    
    for i, j in ti.ndrange(m_g, m_g):

        I = i * m_g + j 

        if is_fluid(i, j):
            if not is_solid(i-1, j):
                A[I, I] += 1.0
                if not is_air(i-1, j):
                    A[I - m_g, I] -= 1.0
            
            if not is_solid(i+1, j):
                A[I, I] += 1.0
                if not is_air(i+1, j):
                    A[I + m_g, I] -= 1.0
            
            if not is_solid(i, j-1):
                A[I, I] += 1.0
                if not is_air(i, j-1):
                    A[I, I - 1] -= 1.0
                
            if not is_solid(i, j+1):
                A[I, I] += 1.0
                if not is_air(i, j+1):
                    A[I, I + 1] -= 1.0
        else:
            #if is_solid(i, j) or is_air(i, j)
            A[I, I] += 1.0
            F_b[I] = 0


@ti.kernel
def copy_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * m_g + I[1]]


frame = 0
def step():

    global frame
    frame += 1
    print("frame", frame)

    advect_particles()


    init_step()

    particle_to_grid()
    grid_normalization()

    last_velocities.copy_from(velocities)

    apply_gravity()
    handle_boundary()


    solve_divergence()


    # sparse matrix solver
    N = n_grid
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    F_b = ti.ndarray(ti.f32, shape=N)
    fill_matrix(K, F_b)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)
    p = solver.solve(F_b)
    copy_pressure(p, pressures)

    # # jacobian iteration
    # for i in range(jacobi_iters):
    # 	global pressures, new_pressures
    # 	pressure_jacobi(pressures, new_pressures)
    # 	pressures, new_pressures = new_pressures, pressures

    projection()


    grid_to_particle()



init_grid()
init_particle()


gui = ti.GUI("FLIP Blending", (res, res))

pause = False
result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)


for frame in range(45000):

    gui.clear(0xFFFFFF)


    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == 'w':
            pause = True
        elif e.key == 'e':
            pause = False
        elif e.key == 'd':
            debug = True
        elif e.key == 'f':
            debug = False

    if not pause:
        step()
    # import time
    # time.sleep(1)
    if debug:
        if frame > 5:
            pause=True


    # break
    if debug:
        for i in range(m_g):
            for j in range(m_g):
                color = 0
                if types[i, j] == FLUID:
                    color = 0xFFFFFF
                elif types[i, j] == AIR:
                    color = 0x0000FF
                elif types[i, j] == SOLID:
                    color = 0xFF0000
                gui.circle([(i+0.5)/m_g, (j+0.5)/m_g], radius = 2, color = color)
                gui.line([i*dx, j*dx], [i*dx, (j+1)*dx], color = 0xFF0000)
                gui.line([i*dx, (j+1)*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
                gui.line([(i+1)*dx, j*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
                gui.line([(i+1)*dx, j*dx], [i*dx, j*dx], color = 0xFF0000)
                gui.text(f'{pressures[i, j]:.2f}p {types[i, j]}t', pos=((i+0.5)/m_g - dx * 0.25, (j+0.75)/m_g), color=0xFF0000)
                gui.text(f'({velocities_before_projection[i, j].x:.2f}, {velocities_before_projection[i, j].y:.2f})v0', pos=((i+0.5)/m_g - dx * 0.25, (j+0.45)/m_g), color=0x000000)
                gui.text(f'({velocities[i, j].x:.2f}, {velocities[i, j].y:.2f})v', pos=((i+0.5)/m_g - dx * 0.25, (j+0.25)/m_g), color=0x000000)

        for i in range(neighbor_count[0]):
            gui.circle([(neighbors[i].x+0.5)*dx, (neighbors[i].y+0.5)*dx], radius = 4, color = 0)
        gui.circle([particle_position[pid_to_check].x, particle_position[pid_to_check].y], radius = 4, color = 0xFF0000)
        i = neighbors[4].x
        j = neighbors[4].y
        gui.line([i*dx, j*dx], [i*dx, (j+1)*dx], color = 0)
        gui.line([i*dx, (j+1)*dx], [(i+1)*dx, (j+1)*dx], color = 0)
        gui.line([(i+1)*dx, j*dx], [(i+1)*dx, (j+1)*dx], color = 0)
        gui.line([(i+1)*dx, j*dx], [i*dx, j*dx], color = 0)
    gui.circles(particle_position.to_numpy() / length, radius=2.8, color=0x3399FF)

    # gui.text('FLIP Blending', pos=(0.05, 0.95), color=0x0)

    # video_manager.write_frame(gui.get_image())	
    gui.show()



# video_manager.make_video(gif=True, mp4=True)