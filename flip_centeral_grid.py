from typing import List
import taichi as ti
import numpy as np
import time

from stable_fluid import K


ti.init(arch=ti.cuda, default_fp=ti.f32, debug=False)


res = 512 * 1
# dt = 1.6e-2 #2e-3 #2e-2
dt = 0.4e-2


rho = 1
jacobi_iters = 6
jacobi_damped_para = 0.76
FLIP_blending = 0.0



m_g = 128
n_grid = m_g*m_g
n_particle = n_grid*4

length = 1.0
dx = length/m_g
inv_dx = 1/dx
substep_num = 1
dt /= substep_num

# solid boundary
boundary_width = 1

eps = 1e-5

debug = False
pause = False

DAM_BREAK = 0
CENTER_DAM_BREAK = 1
LEFT_VERTICAL_STRIP = 2
CENTER_VERTICAL_STRIP = 3
example_case = CENTER_DAM_BREAK

use_jacobi = True

# use_weight = False

use_extrapolation = True

gravity = ti.Vector([0.0, -9.8])

record_video = False
video_name = "video"
video_fps = 30
video_frame = 450

# MAC grid
visit_old_x = ti.field(dtype=ti.i32, shape=(m_g, m_g))
visit_old_y = ti.field(dtype=ti.i32, shape=(m_g, m_g))
visit_x = ti.field(dtype=ti.i32, shape=(m_g, m_g))
visit_y = ti.field(dtype=ti.i32, shape=(m_g, m_g))
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
def init_solid():
    for i, j in types:
        if i < boundary_width or i >= m_g-boundary_width or j < boundary_width or j >= m_g-boundary_width:
            types[i, j] = SOLID


# should not generate particles in solid cells
@ti.kernel
def init_particle():
    for i in particle_position:
        if example_case == DAM_BREAK:
            particle_position[i] = (ti.Vector([ti.random(), ti.random()])*0.5 + ti.Vector([0.05, 0.05])) * length
            # particle_position[i] = (ti.Vector([ti.random()*(1-12*dx), ti.random()*dx*5]) + ti.Vector([2*dx, 2*dx])) * length
            # particle_position[i] = (ti.Vector([ti.random()*0.2, ti.random()*0.3]) + ti.Vector([0.25, 0.25])) * length
        elif example_case == CENTER_DAM_BREAK:
            particle_position[i] = (ti.Vector([ti.random(), ti.random()])*0.5 + ti.Vector([0.3, 0.3])) * length
        elif example_case == LEFT_VERTICAL_STRIP:
            particle_position[i] = (ti.Vector([ti.random()*3*dx, ti.random()*length-dx*boundary_width-dx*boundary_width - 0*dx]) + ti.Vector([dx*boundary_width, dx*boundary_width]))
        elif example_case == CENTER_VERTICAL_STRIP:
            particle_position[i] = (ti.Vector([ti.random()*3*dx, ti.random()*length-dx*boundary_width-dx*boundary_width - 0*dx]) + ti.Vector([0.5*length-dx, dx*boundary_width]))
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


# @ti.kernel
# def handle_boundary():
#     for i, j in velocities:
#         if is_solid(i, j):
#             velocities[i, j] = [0.0, 0.0]

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
    
    for i, j in visit_x:
        visit_x[i, j] = 0
        visit_y[i, j] = 0


@ti.func
def scatter(grid_v, grid_m, xp, vp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * vp
            grid_m[base + offset] += weight


@ti.kernel
def particle_to_grid():

    for k in particle_position:

        pos = particle_position[k]
        vel = particle_velocity[k]

        stagger = ti.Vector([0.5, 0.5])

        scatter(velocities, weights, pos, vel, stagger)


@ti.kernel
def grid_normalization():

    for k in ti.grouped(velocities):
        weight = weights[k]
        if weight > 0:
            velocities[k] = velocities[k] / weight



@ti.kernel
def apply_gravity():
    for i, j in velocities:
        # Is this better? No
        if not is_solid(i, j):
            velocities[i, j] += gravity * dt


@ti.kernel
def adjust_grid_type():
    for i, j in velocities:
        if is_solid(i, j):
            if is_fluid(i-1, j) and velocities[i, j].x <= 0.0:
                types[i, j] = AIR
            if is_fluid(i+1, j) and velocities[i, j].x >= 0.0:
                types[i, j] = AIR
            if is_fluid(i, j-1) and velocities[i, j].y <= 0.0:
                types[i, j] = AIR
            if is_fluid(i, j+1) and velocities[i, j].y >= 0.0:
                types[i, j] = AIR


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


@ti.func
def sample_vel_grad_u(i, j):
    vl = 0.0
    if not is_solid(i-1, j):
        vl = 0.5*(velocities[i-1, j] + velocities[i, j]).x
    else:
        vl = 0.0
    
    vr = 0.0
    if not is_solid(i+1, j):
        vr = 0.5*(velocities[i+1, j] + velocities[i, j]).x
    else:
        vr = 0.0
    return (vr - vl) / dx


@ti.func
def sample_vel_grad_v(i, j):
    vd = 0.0
    if not is_solid(i, j-1):
        vd = 0.5*(velocities[i, j-1] + velocities[i, j]).y
    else:
        vd = 0.0
    
    vu = 0.0
    if not is_solid(i+1, j):
        vu = 0.5*(velocities[i, j+1] + velocities[i, j]).y
    else:
        vu = 0.0
    return (vu - vd) / dx


@ti.kernel
def solve_divergence():

    for i, j in divergences:
        # if weights[i, j] > 0 and not is_solid(i, j):
        if is_fluid(i, j):

            v_l = velocities[i-1, j].x# * weights[i-1, j]
            v_r = velocities[i+1, j].x# * weights[i+1, j]
            v_d = velocities[i, j-1].y# * weights[i, j-1]
            v_u = velocities[i, j+1].y# * weights[i, j+1]
            factor = 1.0
            if is_solid(i-1, j): 
                v_l = -velocities[i, j].x * factor# * weights[i, j]
            if is_solid(i+1, j): 
                v_r = -velocities[i, j].x * factor# * weights[i, j]
            if is_solid(i, j-1): 
                v_d = -velocities[i, j].y * factor# * weights[i, j]
            if is_solid(i, j+1): 
                v_u = -velocities[i, j].y * factor# * weights[i, j]
            div = (v_r - v_l + v_u - v_d) / (dx*2)
            divergences[i, j] = div

            # divergences[i, j] = sample_vel_grad_u(i, j) + sample_vel_grad_v(i, j)


@ti.kernel
def pressure_jacobi(p:ti.template(), new_p:ti.template()):

    w = jacobi_damped_para

    for i, j in p:
        if is_fluid(i, j):
        # if weights[i, j] > 0 and not is_solid(i, j):

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
        else:
            new_p[i, j] = 0.0


@ti.kernel
def fill_matrix(A: ti.types.sparse_matrix_builder(), F_b: ti.types.ndarray()):
    
    for I in ti.grouped(divergences):
        F_b[I[0] * m_g + I[1]] = - divergences[I] * dx * dx * rho / dt
    
    for i, j in ti.ndrange(m_g, m_g):
        I = i * m_g + j
        # if weights[i, j] > 0 and not is_solid(i, j):
        if is_fluid(i, j):
            if not is_solid(i-1, j):
                # volume = 0.5*(weights[i, j] + weights[i-1, j])
                volume = 1.0
                A[I, I] += 1.0 * volume
                if not is_air(i-1, j):
                    A[I - m_g, I] -= 1.0 * volume
            
            if not is_solid(i+1, j):
                # volume = 0.5*(weights[i, j] + weights[i+1, j])
                volume = 1.0
                A[I, I] += 1.0 * volume
                if not is_air(i+1, j):
                    A[I + m_g, I] -= 1.0 * volume
            
            if not is_solid(i, j-1):
                # volume = 0.5*(weights[i, j] + weights[i, j-1])
                volume = 1.0
                A[I, I] += 1.0 * volume
                if not is_air(i, j-1):
                    A[I, I - 1] -= 1.0 * volume
                
            if not is_solid(i, j+1):
                # volume = 0.5*(weights[i, j] + weights[i, j+1])
                volume = 1.0
                A[I, I] += 1.0 * volume
                if not is_air(i, j+1):
                    A[I, I + 1] -= 1.0 * volume
        else:
            #if is_solid(i, j) or is_air(i, j)
            A[I, I] += 1.0
            F_b[I] = 0


@ti.kernel
def copy_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * m_g + I[1]]


@ti.kernel
def projection():
    for i, j in ti.ndrange(m_g, m_g):
        velocities_before_projection[i, j] = velocities[i, j]
        # if weights[i, j] > 0 and not is_solid(i, j): 
        if is_fluid(i, j):
        # if not is_solid(i, j):
            # grad_p = ti.Vector([pressures[i+1, j]-pressures[i-1, j], pressures[i, j+1]-pressures[i, j-1]]) / (dx*2)
            grad_p = ti.Vector([pressures[i+1, j]-pressures[i, j], pressures[i, j+1]-pressures[i, j]]) / dx
            # if is_solid(i-1, j):
            #     grad_p.x = (pressures[i+1, j]-pressures[i, j])/dx
            if is_solid(i+1, j):
                grad_p.x = (pressures[i, j]-pressures[i-1, j])/dx
            # if is_solid(i, j-1):
            #     grad_p.y = (pressures[i, j+1]-pressures[i, j])/dx
            if is_solid(i, j+1):
                grad_p.y = (pressures[i, j]-pressures[i, j-1])/dx
            velocities[i, j] -= grad_p / rho * dt
            visit_x[i, j] = 1
            visit_y[i, j] = 1

        # # air cells. Is this better? No if velocity extrapolation enables.
        # elif is_air(i, j):
        #     grad_p = ti.Vector([pressures[i+1, j]-pressures[i, j], pressures[i, j+1]-pressures[i, j]]) / dx
        #     if is_fluid(i+1, j):
        #         velocities[i, j].x -= grad_p.x / rho * dt
        #         visit_x[i, j] = 1
        #     elif is_fluid(i-1, j):
        #         grad_p.x = (pressures[i, j]-pressures[i-1, j])/dx
        #         velocities[i, j].x -= grad_p.x / rho * dt
        #         visit_x[i, j] = 1
        #     else:
        #         pass
        #         # visit_x[i, j] = 1

        #     if is_fluid(i, j+1):
        #         velocities[i, j].y -= grad_p.y / rho * dt
        #         visit_y[i, j] = 1
        #     elif is_fluid(i, j-1):
        #         grad_p.y = (pressures[i, j]-pressures[i, j-1])/dx
        #         velocities[i, j].y -= grad_p.y / rho * dt
        #         visit_y[i, j] = 1
        #     else:
        #         pass


    for i, j in ti.ndrange(m_g, m_g):
        if is_solid(i, j):
            if not is_solid(i-1, j):
                velocities[i, j].x = - velocities[i-1, j].x
                visit_x[i, j] = 1
                # visit_y[i, j] = 1
            if not is_solid(i+1, j):
                velocities[i, j].x = - velocities[i+1, j].x
                visit_x[i, j] = 1
                # visit_y[i, j] = 1
            if not is_solid(i, j-1):
                velocities[i, j].y = - velocities[i, j-1].y
                # visit_x[i, j] = 1
                visit_y[i, j] = 1
            if not is_solid(i, j+1):
                velocities[i, j].y = - velocities[i, j+1].y
                # visit_x[i, j] = 1
                visit_y[i, j] = 1


    # for i, j in ti.ndrange(m_g, m_g):
    #     if is_solid(i, j):
    #         if not is_solid(i-1, j):
    #             velocities[i, j] = - velocities[i-1, j]
    #             visit_x[i, j] = 1
    #             visit_y[i, j] = 1
    #         if not is_solid(i+1, j):
    #             velocities[i, j] = - velocities[i+1, j]
    #             visit_x[i, j] = 1
    #             visit_y[i, j] = 1
    #         if not is_solid(i, j-1):
    #             velocities[i, j] = - velocities[i, j-1]
    #             visit_x[i, j] = 1
    #             visit_y[i, j] = 1
    #         if not is_solid(i, j+1):
    #             velocities[i, j] = - velocities[i, j+1]
    #             visit_x[i, j] = 1
    #             visit_y[i, j] = 1

    # for i, j in ti.ndrange(m_g, m_g):
    #     # let the velocity of the wall be the negative velocity of it's nearest water cell to ensure the (average) velocity on the boundary in zero.
    #     # if is_solid(i, j) and weights[i, j] > 0:
    #     if is_solid(i, j):
    #         if is_fluid(i+1, j) and velocities[i+1, j].x < 0:
    #             velocities[i, j].x = -velocities[i+1, j].x
    #         elif is_fluid(i-1, j) and velocities[i-1, j].x > 0:
    #             velocities[i, j].x = -velocities[i-1, j].x
    #         else:
    #             velocities[i, j].x = 0.0
            
    #         if is_fluid(i, j+1) and velocities[i, j+1].y < 0:
    #             velocities[i, j].y = -velocities[i, j+1].y
    #         elif is_fluid(i, j-1) and velocities[i, j-1].y > 0:
    #             velocities[i, j].y = -velocities[i, j-1].y
    #         else:
    #             velocities[i, j].y = 0.0


@ti.kernel
def extrapolate_velocity():
    ti.loop_config(serialize=True)
    for i in range(m_g):
        for j in range(m_g):
            if not visit_old_x[i, j]:
                count = 0
                sum = 0.0
                if i > 0 and visit_old_x[i-1, j] == 1:
                    count += 1
                    sum += velocities[i-1, j].x
                if i < m_g-1 and visit_old_x[i+1, j] == 1:
                    count += 1
                    sum += velocities[i+1, j].x
                if j > 0 and visit_old_x[i, j-1] == 1:
                    count += 1
                    sum += velocities[i, j-1].x
                if j < m_g-1 and visit_old_x[i, j+1] == 1:
                    count += 1
                    sum += velocities[i, j+1].x
                if count > 0:
                    velocities[i, j].x = sum / count
                    visit_x[i, j] = 1

    ti.loop_config(serialize=True)
    for i in range(m_g):
        for j in range(m_g):
            if not visit_old_y[i, j]:
                count = 0
                sum = 0.0
                if i > 0 and visit_old_y[i-1, j] == 1:
                    count += 1
                    sum += velocities[i-1, j].y
                if i < m_g-1 and visit_old_y[i+1, j] == 1:
                    count += 1
                    sum += velocities[i+1, j].y
                if j > 0 and visit_old_y[i, j-1] == 1:
                    count += 1
                    sum += velocities[i, j-1].y
                if j < m_g-1 and visit_old_y[i, j+1] == 1:
                    count += 1
                    sum += velocities[i, j+1].y
                if count > 0:
                    velocities[i, j].y = sum / count
                    visit_y[i, j] = 1

@ti.func
def gather(grid_v, last_grid_v, xp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

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

    stagger = ti.Vector([0.5, 0.5])
    
    for k in ti.grouped(particle_position):
    
        p = particle_position[k]

        pic_x, flip_dx = gather(velocities, last_velocities, p, stagger)

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


        particle_velocity[k] = vel
        particle_position[k] = pos


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
    # handle_boundary()
    # adjust_grid_type()




    solve_divergence()

    if not use_jacobi:
        # sparse matrix solver
        N = n_grid
        K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
        F_b = ti.ndarray(ti.f32, shape=N)
        fill_matrix(K, F_b)
        L = K.build()
        # K.print_triplets()
        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(L)
        solver.factorize(L)
        p = solver.solve(F_b)
        global pressures
        copy_pressure(p, pressures)
    else:
        # jacobian iteration
        for i in range(jacobi_iters):
            global new_pressures
            pressure_jacobi(pressures, new_pressures)
            pressures, new_pressures = new_pressures, pressures



    # init_solid()
    projection()

    
    if use_extrapolation:
        for i in range(2):
            visit_old_x.copy_from(visit_x)
            visit_old_y.copy_from(visit_y)
            extrapolate_velocity()



    grid_to_particle()



init_solid()
init_particle()


gui = ti.GUI("FLIP Blending", (res, res))

result_dir = "./result"
video_manager = None
if record_video:
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=video_fps, automatic_build=False, video_filename=video_name)

for frame in range(video_frame if record_video else 4500000):

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
        for i in range(substep_num):
            step()
    # import time
    # time.sleep(1)
    # if debug:
    #     if frame > 5:
    #         pause=True


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
                # gui.line([(i+1)*dx, j*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
                # gui.line([(i+1)*dx, j*dx], [i*dx, j*dx], color = 0xFF0000)
                # gui.text(f'{weights[i, j]:.2f}w {types[i, j]}t', pos=((i+0.5)/m_g - dx * 0.25, (j+0.85)/m_g), color=0x00FF00)
                gui.text(f'{pressures[i, j]:.2f}p {types[i, j]}t', pos=((i+0.5)/m_g - dx * 0.25, (j+0.75)/m_g), color=0xFF0000)
                # gui.text(f'({velocities_before_projection[i, j].x:.2f}, {velocities_before_projection[i, j].y:.2f})v0', pos=((i+0.5)/m_g - dx * 0.25, (j+0.45)/m_g), color=0x000000)
                gui.text(f'({velocities[i, j].x:.2f}, {velocities[i, j].y:.2f})v', pos=((i+0.5)/m_g - dx * 0.25, (j+0.25)/m_g), color=0x000000)
                # gui.text(f'({velocities_before_projection[i, j].y:.2f})v0', pos=((i+0.5)/m_g - dx * 0.25, (j+0.45)/m_g), color=0x000000)
                # gui.text(f'({velocities[i, j+1].y:.2f})', pos=((i+0.5)/m_g - dx * 0.25, (j+0.45)/m_g), color=0x000000)
                # gui.text(f'({velocities[i, j].x:.2f})v', pos=((i+0.5)/m_g - dx * 0.25, (j+0.25)/m_g), color=0x000000)
    gui.circles(particle_position.to_numpy() / length, radius=1.0*(res//512), color=0x3399FF)


    if record_video:
        pos = [0.05, 0.95]
        offset = [0.0, 0.05]
        gui.text(f'* Gravity: {gravity.y}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* Time step: {dt}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* Subtemp Num: {substep_num}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* Grid Res: {m_g}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* Boundary Size: {boundary_width}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* Velocity Extrapolation: {use_extrapolation}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* Use Jacobi Iteration: {use_jacobi}', pos=pos, color=0x0, font_size=(res//512)*15)
        if use_jacobi:
            pos[1] -= offset[1]
            gui.text(f'  - Iteration Num: {jacobi_iters}', pos=pos, color=0x0, font_size=(res//512)*15)
            pos[1] -= offset[1]
            gui.text(f'  - Damping Para: {jacobi_damped_para}', pos=pos, color=0x0, font_size=(res//512)*15)
        pos[1] -= offset[1]
        gui.text(f'* FLIP Blending: {FLIP_blending}', pos=pos, color=0x0, font_size=(res//512)*15)
        video_manager.write_frame(gui.get_image())	
    gui.show()



if record_video:
    video_manager.make_video(gif=True, mp4=True)