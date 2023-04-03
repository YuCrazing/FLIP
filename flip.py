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
dt = 2e-2 #2e-3 #2e-2


rho = 1000
jacobi_iters = 300
jacobi_damped_para = 1
FLIP_blending = 0.0



m_g = 64
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
velocities_u = ti.field(dtype=ti.f32, shape=(m_g+1, m_g))
velocities_v = ti.field(dtype=ti.f32, shape=(m_g, m_g+1))

last_velocities_u = ti.field(dtype=ti.f32, shape=(m_g+1, m_g))
last_velocities_v = ti.field(dtype=ti.f32, shape=(m_g, m_g+1))

weights_u = ti.field(dtype=ti.f32, shape=(m_g+1, m_g))
weights_v = ti.field(dtype=ti.f32, shape=(m_g, m_g+1))


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
        particle_position[i] = (ti.Vector([ti.random(), ti.random()])*0.5 + 0.05) * length
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

    for i, j in velocities_u:
        if is_solid(i-1, j) or is_solid(i, j):
            velocities_u[i, j] = 0.0	
    
    for i, j in velocities_v:
        if is_solid(i, j-1) or is_solid(i, j):
            velocities_v[i, j] = 0.0

@ti.kernel
def init_step():

    for i, j in types:
        if not is_solid(i, j):
            types[i, j] = AIR

    for k in particle_velocity:
        grid = (particle_position[k] * inv_dx).cast(int)
        if not is_solid(grid.x, grid.y):
            types[grid] = FLUID


    for k in ti.grouped(velocities_u):
        velocities_u[k] = 0.0
        weights_u[k] = 0.0
    
    for k in ti.grouped(velocities_v):
        velocities_v[k] = 0.0
        weights_v[k] = 0.0

    for k in ti.grouped(pressures):
        if is_air(k.x, k.y):
            pressures[k] = 0.0
            new_pressures[k] = 0.0


@ti.func
def scatter(grid_v, grid_m, xp, vp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * vp
            grid_m[base + offset] += weight


@ti.kernel
def particle_to_grid():

    for k in particle_position:

        pos = particle_position[k]
        vel = particle_velocity[k]

        stagger_u = ti.Vector([0.0, 0.5])
        stagger_v = ti.Vector([0.5, 0.0])

        scatter(velocities_u, weights_u, pos, vel.x, stagger_u)
        scatter(velocities_v, weights_v, pos, vel.y, stagger_v)


@ti.kernel
def grid_normalization():

    for k in ti.grouped(velocities_u):
        weight = weights_u[k]
        if weight > 0:
            velocities_u[k] = velocities_u[k] / weight

    for k in ti.grouped(velocities_v):
        weight = weights_v[k]
        if weight > 0:
            velocities_v[k] = velocities_v[k] / weight


@ti.kernel
def apply_gravity():
    for i, j in velocities_v:
        # if not is_solid(i, j-1) and not is_solid(i, j):
            velocities_v[i, j] += -9.8 * dt


@ti.kernel
def solve_divergence():

    for i, j in divergences:
        if not is_solid(i, j):

            v_l = velocities_u[i, j]
            v_r = velocities_u[i+1, j]
            v_d = velocities_v[i, j]
            v_u = velocities_v[i, j+1]

            div = v_r - v_l + v_u - v_d

            if is_solid(i-1, j): 
                div += v_l
            if is_solid(i+1, j): 
                div -= v_r
            if is_solid(i, j-1): 
                div += v_d
            if is_solid(i, j+1): 
                div -= v_u

            divergences[i, j] = div / (dx)


@ti.kernel
def pressure_jacobi(p:ti.template(), new_p:ti.template()):

    w = jacobi_damped_para

    for i, j in p:
        if is_fluid(i, j):

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
            F_b[I] = 0.0




@ti.kernel
def copy_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * m_g + I[1]]

@ti.kernel
def projection():

    for i, j in ti.ndrange(m_g+1, m_g):
        if is_fluid(i-1, j) or is_fluid(i, j):
            if is_solid(i-1, j) or is_solid(i, j):
                velocities_u[i, j] = 0.0
            else:
                velocities_u[i, j] -= (pressures[i, j] - pressures[i-1, j]) / dx / rho * dt

    for i, j in ti.ndrange(m_g, m_g+1):
        if is_fluid(i, j-1) or is_fluid(i, j):
            if is_solid(i, j-1) or is_solid(i, j):
                velocities_v[i, j] = 0.0
            else:
                velocities_v[i, j] -= (pressures[i, j] - pressures[i, j-1]) / dx / rho * dt


@ti.func
def gather(grid_v, last_grid_v, xp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    v_pic = 0.0
    v_flip = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            v_pic  += weight * grid_v[base + offset]
            v_flip += weight * (grid_v[base + offset] - last_grid_v[base + offset])

    return v_pic, v_flip


@ti.kernel
def grid_to_particle():

    stagger_u = ti.Vector([0.0, 0.5])
    stagger_v = ti.Vector([0.5, 0.0])
    
    for k in ti.grouped(particle_position):
    
        p = particle_position[k]

        pic_x, flip_dx = gather(velocities_u, last_velocities_u, p, stagger_u)
        pic_y, flip_dy = gather(velocities_v, last_velocities_v, p, stagger_v)

        pic_vel = ti.Vector([pic_x, pic_y])
        flip_vel = particle_velocity[k] + ti.Vector([flip_dx, flip_dy])

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


frame = 0
def step():

    global frame
    frame += 1
    print("frame", frame)

    init_step()

    particle_to_grid()
    grid_normalization()

    last_velocities_u.copy_from(velocities_u)
    last_velocities_v.copy_from(velocities_v)

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

    # jacobian iteration
    # for i in range(jacobi_iters):
    # 	global pressures, new_pressures
    # 	pressure_jacobi(pressures, new_pressures)
    # 	pressures, new_pressures = new_pressures, pressures

    projection()


    grid_to_particle()
    advect_particles()



init_grid()
init_particle()


gui = ti.GUI("FLIP Blending", (res, res))


# result_dir = "./result"
# video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

pause = False

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
        for i in range(m_g):
            for j in range(m_g+1):
                gui.text(f'({velocities_v[i, j]:.2f})v', pos=((i+0.5)/m_g - dx * 0.25, (j)/m_g + dx * 0.25), color=0x000000)
    gui.circles(particle_position.to_numpy() / length, radius=1.8, color=0x3399FF)

    gui.text('FLIP Blending', pos=(0.05, 0.95), color=0x0)

    # video_manager.write_frame(gui.get_image())	
    gui.show()



# video_manager.make_video(gif=True, mp4=True)