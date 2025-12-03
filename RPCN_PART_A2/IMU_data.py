import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load IMU data
df = pd.read_csv('RPCN_PART_A2/data/imu_data_1.csv', sep=',', comment='/', engine='python')
# print(df.columns)
# 4. Plot X, Y position


# 2. Apply scale factors (replace with your actual values)
bx = 0.060675  # Example scale factor for X
by = -0.035453 # Example scale factor for Y
Sx = 0.000102   # Example bias for X
Sy = 0.000763  # Example bias for Y
Sw = -2.318470e-03
bw = -7.373079e-01

dt = 0.002502  # Sampling interval in seconds

# # Corrected acceleration
# acc_x = (1 + Sx) * df['field.linear_acceleration.x'] + bx
# acc_y = (1 + Sy) * df['field.linear_acceleration.y'] + by 
# rewrite to go from raw to corrected
acc_x = (df['field.linear_acceleration.x'] - bx) / (1 + Sx)
acc_y = (df['field.linear_acceleration.y'] - by) / (1 + Sy)
omega_z = (df['field.angular_velocity.z'] - bw) / (1 + Sw)


# --- Rotation correction around Z axis ---
A = np.cumsum(omega_z) * dt  # Integrate angular velocity to get angle (using sampling interval)

# create rotation matrix for each time step
cos_A = np.cos(A)
sin_A = np.sin(A)
R = np.array([[cos_A, -sin_A],
              [sin_A, cos_A]]).transpose((2, 0, 1))  # Shape (N, 2, 2)

# Rotate accelerations
acc_corrected = np.einsum('ijk,ik->ij', R, np.vstack((acc_x, acc_y)).T)
acc_x_rot = acc_corrected[:, 0]
acc_y_rot = acc_corrected[:, 1]
# 3. Integrate accelerations to get velocities and positions
vel_x = np.cumsum(acc_x_rot) * dt
vel_y = np.cumsum(acc_y_rot) * dt
pos_x = np.cumsum(vel_x) * dt
pos_y = np.cumsum(vel_y) * dt
# 4. Plot X, Y position
plt.figure()
plt.plot(pos_x, pos_y)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Position Plot')
plt.axis('equal')
# plt.show()

# # unbiased version for comparison
# acc_x = (df['field.linear_acceleration.x'])
# acc_y = (df['field.linear_acceleration.y'])
# omega_z = (df['field.angular_velocity.z'])


# # --- Rotation correction around Z axis ---
# A = np.cumsum(omega_z) * dt  # Integrate angular velocity to get angle (using sampling interval)

# # create rotation matrix for each time step
# cos_A = np.cos(A)
# sin_A = np.sin(A)
# R = np.array([[cos_A, -sin_A],
#               [sin_A, cos_A]]).transpose((2, 0, 1))  # Shape (N, 2, 2)

# # Rotate accelerations
# acc_corrected = np.einsum('ijk,ik->ij', R, np.vstack((acc_x, acc_y)).T)
# acc_x_rot = acc_corrected[:, 0]
# acc_y_rot = acc_corrected[:, 1]
# # 3. Integrate accelerations to get velocities and positions
# vel_x = np.cumsum(acc_x_rot) * dt
# vel_y = np.cumsum(acc_y_rot) * dt
# pos_x = np.cumsum(vel_x) * dt
# pos_y = np.cumsum(vel_y) * dt
# # 4. Plot X, Y position

# plt.plot(pos_x, pos_y)

plt.show()




# plot the angle A
# plt.figure()
# plt.plot(df['%time'], A)
# plt.ylabel('Angle (rad)')
# plt.xlabel('Time (s)')
# plt.show()