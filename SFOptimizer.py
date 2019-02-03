#  Copyright 2019 Dave Fernandes. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" Saddle-Free Optimizer """


from enum import Enum
import tensorflow as tf


"""
	Damping options for SFOptimizer
"""
class SFDamping(Enum):
	tikhonov = 1	# Add damping coefficient to diagonal elements of curvature matrix
	marquardt = 2	# Multiply diagonal elements of curvature matrix by (1 + damping coefficient)
	curvature = 3	# Multiply curvature matrix by (1 + damping coefficient)


"""
	Saddle-Free optimizer for Tensorflow
	See: https://arxiv.org/abs/1406.2572
	
	Methods to use:
		__init__: Initialize variables.
			Construct this class before running global_variables_initializer.
			
		reset_lambda: Create op to reset lambda_damp to its initial value
			Create this op after running global_variables_initializer.
		
		minimize: Create op for doing Lanczos iterations and computing updates
			Create this op after running global_variables_initializer.
		
		fixed_subspace_step: Create op for computing updates within existing Krylov subspace
			Create this op after running global_variables_initializer.
		
		update: Create op for update of variables
			Create this op after running global_variables_initializer.
			
		Run the minimize and update ops in separate calls to Session.run in your
		main optimization loop.
"""
class SFOptimizer(object):
	
	"""
		Create variables
			var_list: Python list of TF variables
				List of second-order differentiable variables to optimize
			krylov_dimension: int
				Subspace dimension for Newton step computation
			damping_type: SFDamping
				Style of damping for trust region (see description for SFDamping Enum)
			initial_damping: float
				Initial value for Levenberg-Marquardt damping coefficient
			dtype: Tensorflow type
				Type of Tensorflow variables
	"""
	def __init__(self, var_list, krylov_dimension=20, damping_type=SFDamping.tikhonov, initial_damping=0.01, dtype=tf.float32):
		assert(krylov_dimension > 3)
		self.krylov_dim = krylov_dimension
		self.damping_type = damping_type
		self.initial_damping = initial_damping
		self.dtype = dtype
		self.var_list = var_list
		self.stashed_var_list = []
		
		if dtype == tf.float64:
			self.epsilon = 0.0000000000000002
		else:
			self.epsilon = 0.0000001192
			
		# Thresholds on rho for incrementing/decrementing lambda
		self.rho_decrement_thresh = 0.3
		self.rho_increment_thresh = 0.0
				
		# Tensor containing eigenvalues of the Hessian in the subspace
		self.h_eigenvalues = None
		
		# Tensor containing rho criterion for Levenberg-Marquardt heuristic
		self.rho = None
		
		# Variables
		with tf.name_scope('SFOptimizer'):
			# Damping coefficient
			self.lambda_damp = tf.get_variable("lambda_damp", initializer=tf.constant(initial_damping, dtype=dtype), trainable=False, use_resource=True)
			
			# Loss computed before updating training variables
			self.saved_loss = tf.get_variable("saved_loss", shape=(), initializer=tf.zeros_initializer(), trainable=False, dtype=dtype, use_resource=True)
			
			# Delta loss after a Newton step if the minimization surface had been quadratic
			self.quadratic_loss = tf.get_variable("quadratic_loss", shape=(), initializer=tf.zeros_initializer(), trainable=False, dtype=dtype, use_resource=True)
			
			# Copy of training variables saved before updating them
			self.p_count = 0
			for v in var_list:
				self.stashed_var_list.append(tf.get_variable("stashed_" + v.name.split(':')[0], initializer=v.initialized_value(), trainable=False, use_resource=True))
				self.p_count += v.shape.num_elements()
			
			# Variables for minimization steps in the krylov subspace
			if self.p_count > self.krylov_dim:
				# Previous search direction initialized to random unit vector
				self.w_prev = tf.get_variable("w_prev", initializer=tf.math.l2_normalize(tf.random_uniform([self.p_count, 1], -1, 1, dtype=dtype)), trainable=False, use_resource=True)
				
				# Subspace transform, gradient and hessian (saved between steps)
				self.Q = tf.get_variable("Q", [self.p_count, self.krylov_dim], initializer=tf.zeros_initializer(), trainable=False, dtype=dtype, use_resource=True)
				self.grad_k = tf.get_variable("grad_k", [self.krylov_dim, 1], initializer=tf.zeros_initializer(), trainable=False, dtype=dtype, use_resource=True)
				self.hess_k = tf.get_variable("hess_k", [self.krylov_dim, self.krylov_dim], initializer=tf.zeros_initializer(), trainable=False, dtype=dtype, use_resource=True)
	
	"""
		Return the optimizer's name
	"""
	def get_name(self):
		return 'Saddle-Free'

	"""
		Reset the damping parameter to its initial value
	"""
	def reset_lambda(self):
		return tf.assign(self.lambda_damp, self.initial_damping)

	"""
		Return an operation used to take one minimization step
	"""
	def minimize(self, loss):
		self.loss = loss
		self.gradients = tf.gradients(loss, self.var_list)
		
		# Count parameters to determine whether dimensional reduction is needed; and cache variable values
		cache_ops = []
		for v, sv in zip(self.var_list, self.stashed_var_list):
			cache_ops.append(tf.assign(sv, v))
		
		with tf.control_dependencies(cache_ops):
			if self.p_count > self.krylov_dim:
				minimize_op = self._krylov_minimize()
			else:
				hessians = tf.hessians(loss, self.var_list)
				min_op, dvar_list = self._minimize(self.var_list, self.gradients, hessians)
				
				with tf.control_dependencies([tf.group([min_op])]):
					minimize_op = self._update_variables(dvar_list)
		
		return minimize_op

	"""
		Internal method to return an operation used to take one minimization step
	"""
	def _krylov_minimize(self):
		# First vector is the gradient direction
		gradient_vec = self.var_to_column(self.gradients)
		v_i = tf.math.l2_normalize(gradient_vec)
		
		# Lanczos method follows Paige (1972; variant [2, 7])
		var = self.column_to_var(v_i)
		hv_list = self.hessian_vector_product(var)
		u = self.var_to_column(hv_list)
		
		# Matrix to transform to/from Krylov subspace is concatenation of v_i's
		v = tf.identity(v_i, name="v_transform")
		
		for i in range(self.krylov_dim):
			with tf.control_dependencies([v_i]):
				alpha = tf.matmul(v_i, u, transpose_a=True)
				
				# For the last pass, we only need to append the last row to the transformed Hessian
				if i == self.krylov_dim - 1:
					h_row = tf.concat([tf.zeros([1, self.krylov_dim - 2], dtype=self.dtype), beta_prev, alpha], axis=1)
					h_tridiag = tf.concat([h_tridiag, h_row], axis=0)
					w_assign_op = tf.assign(self.w_prev, v_i)
					break
				
				w = u - alpha * v_i
				beta = tf.reshape(tf.norm(w), [1, 1])
				
				# TODO: Selectively reorthogonalize w to remove numerical drift
				# See: Parlett & Scott (1979)
				
				# Compute next v_i and concatenate onto v
				v_i = w / beta
				v = tf.concat([v, v_i], axis=1)
				
				var = self.column_to_var(v_i)
				hv_list = self.hessian_vector_product(var)
				u = self.var_to_column(hv_list) - beta * tf.slice(v, [0, i], [-1, 1])
				
				# Concatenate a new row to the transformed Hessian
				if i == 0:
					h_tridiag = tf.concat([alpha, beta, tf.zeros([1, self.krylov_dim - 2], dtype=self.dtype)], axis=1)
				else:
					elements = []
					if i > 1:
						elements.append(tf.zeros([1, i - 1], dtype=self.dtype))
					elements.append(beta_prev)
					elements.append(alpha)
					elements.append(beta)
					if i < self.krylov_dim - 2:
						elements.append(tf.zeros([1, self.krylov_dim - i - 2], dtype=self.dtype))
					h_row = tf.concat(elements, axis=1)
					h_tridiag = tf.concat([h_tridiag, h_row], axis=0)
				
				beta_prev = beta
		
		# Transform gradient and parameters to Krylov subspace
		grad_k = tf.matmul(v, gradient_vec, transpose_a=True)
		
		variable_vec = self.var_to_column(self.var_list)
		var_k = tf.matmul(v, variable_vec, transpose_a=True)
		
		# Save subspace data for additional steps
		cache_ops = []
		cache_ops.append(tf.assign(self.grad_k, grad_k))
		cache_ops.append(tf.assign(self.hess_k, h_tridiag))
		cache_ops.append(tf.assign(self.Q, v))
		
		# Minimize in the subspace
		min_op, dvar_list = self._minimize([var_k], [grad_k], [h_tridiag])
		
		# Transform parameters back to original space
		with tf.control_dependencies([tf.group([min_op])]):
			dv_subspace = self.var_to_column(dvar_list)
			dv_full = tf.matmul(v, dv_subspace)
			dvar_full_list = self.column_to_var(dv_full)
			
			# Update the original variables
			update_op = self._update_variables(dvar_full_list)
		
		return tf.group([w_assign_op, update_op] + cache_ops)

	"""
		Take one minimization step inside the krylov subspace without redoing the Lanczos decomp
	"""
	def fixed_subspace_step(self):
		if self.p_count <= self.krylov_dim:
			return tf.print("Subspace descent is not being used")
		
		cache_ops = []
		for v, sv in zip(self.var_list, self.stashed_var_list):
			cache_ops.append(tf.assign(sv, v))
		
		variable_vec = self.var_to_column(self.var_list)
		var_k = tf.matmul(self.Q, variable_vec, transpose_a=True)
		
		# Minimize in the subspace
		min_op, dvar_list = self._minimize([var_k], [self.grad_k], [self.hess_k])
		
		# Transform parameters back to original space
		with tf.control_dependencies([tf.group([min_op])]):
			dv_subspace = self.var_to_column(dvar_list)
			dv_full = tf.matmul(self.Q, dv_subspace)
			dvar_full_list = self.column_to_var(dv_full)
			
			# Update the original variables
			update_op = self._update_variables(dvar_full_list)
		
		return tf.group(cache_ops + [update_op])

	"""
		Internal method to return an operation used to take one minimization step
	"""
	def _minimize(self, var_list, gradients, hessians):
		init_op = tf.assign(self.quadratic_loss, 0.0)
		
		dvar_list = []
		q_loss_ops = []
		self.h_eigenvalues = None
		
		# For each gradient and Hessian in list, perform a step of Newton's method
		with tf.control_dependencies([init_op]):
			for v, g, h in zip(var_list, gradients, hessians):
				# Reshape gradient into vector and Hessian into matrix
				g_reg = tf.reshape(g, [1, tf.size(v)])
				h_reg = tf.reshape(h, [tf.size(v), tf.size(v)])
				
				# Eigen-decomposition of Hessian
				e_val, e_vec = tf.linalg.eigh(h_reg)
				
				if self.h_eigenvalues == None:
					self.h_eigenvalues = e_val
				else:
					self.h_eigenvalues = tf.concat([self.h_eigenvalues, e_val], axis=0)
				
				# Reconstitute Hessian using absolute value of eigenvalues
				h_abs = tf.matmul(tf.matmul(e_vec, tf.diag(tf.abs(e_val))), tf.transpose(e_vec))
				
				# Add damping term to diagonal
				lambda_damp = self.lambda_damp.read_value()
				
				if self.damping_type == SFDamping.tikhonov:
					h_adj = h_abs + tf.multiply(lambda_damp, tf.eye(tf.size(v), dtype=self.dtype))
				elif self.damping_type == SFDamping.marquardt:
					h_adj = h_abs + tf.multiply(lambda_damp, tf.linalg.diag(tf.linalg.diag_part(h_abs)))
				elif self.damping_type == SFDamping.curvature:
					h_adj = h_abs * (1 + lambda_damp)
				
				# Compute Newton step
				h_inv = tf.matrix_inverse(h_adj)
				step = -tf.matmul(g_reg, h_inv)
				
				# Reshape step into shape of original variable
				dv = tf.reshape(step, tf.shape(v))
				dvar_list.append(dv)
				
				# Compute the delta loss for step if surface were a perfect quadratic
				v_reg = tf.reshape(v, [tf.size(v), 1])
				q_loss = tf.matmul(g_reg, v_reg) + 0.5*tf.matmul(tf.transpose(v_reg), tf.matmul(h_adj, v_reg))
				q_loss_ops.append(tf.assign_add(self.quadratic_loss, tf.reshape(q_loss, []), use_locking=True))
		
		return (tf.group(q_loss_ops), dvar_list)
	
	"""
		Internal method to create an operation to update the training variables.
	"""
	def _update_variables(self, dvar_list):
		update_ops = []
		
		for v, dv in zip(self.var_list, dvar_list):
			update_ops.append(tf.assign_add(v, dv, use_locking=True))
		
		update_ops.append(tf.assign(self.saved_loss, self.loss))
		return tf.group(update_ops)
	
	"""
		Compute loss after the training update and adjust lambda_damp if necessary.
		Reverts training variables if loss has increased. The next training step will
		try again with a new value of lambda_damp that will hopefully do better.
	"""
	def update(self):
		def increment_lambda():
			return tf.assign(self.lambda_damp, tf.minimum(self.lambda_damp * 10., 10.0))
		
		def decrement_lambda():
			return tf.assign(self.lambda_damp, tf.maximum(self.lambda_damp * 0.1, 0.00001))
		
		def revert_vars():
			copy_ops = []
			
			for v, sv in zip(self.var_list, self.stashed_var_list):
				copy_ops.append(tf.assign(v, sv))
			
			return tf.group(copy_ops)
		
		adjust_ops = []
		delta_loss = tf.subtract(self.saved_loss, self.loss)
		
		# Levenberg-Marquardt heuristic for updating lambda
		with tf.control_dependencies([delta_loss]):
			# Ratio of actual delta loss after taking step to ideal quadratic loss
			self.rho = tf.divide(delta_loss, tf.abs(self.quadratic_loss))
			
			# If Newton step was too big and loss increased, revert variables to old values
			adjust_ops.append(tf.cond(delta_loss < 0.0, revert_vars, lambda: tf.group([tf.identity(self.lambda_damp)])))
			
			# Adjust lambda if rho is too big or too small
			adjust_ops.append(tf.cond(self.rho > self.rho_decrement_thresh, decrement_lambda, lambda: tf.identity(self.lambda_damp)))
			adjust_ops.append(tf.cond(delta_loss < self.rho_increment_thresh, increment_lambda, lambda: tf.identity(self.lambda_damp)))
		
		return tf.group(adjust_ops)
		
	"""
		Convert a column vector to a list of variables.
	"""
	def column_to_var(self, vec):
		list = []
		offset = 0
		
		for var in self.var_list:
			slice = tf.slice(vec, [offset, 0], [tf.size(var), 1])
			if offset == 0:
				offset = tf.size(var)
			else:
				offset = tf.add(offset, tf.size(var))
			
			list.append(tf.reshape(slice, tf.shape(var)))
		
		return list
		
	"""
		Convert a list of variables to a column vector.
	"""
	def var_to_column(self, var_list):
		flattened_list = []
		for var in var_list:
			flattened_list.append(tf.reshape(var, [tf.size(var), 1]))
		
		vec = tf.concat(flattened_list, axis=0)
		return vec
		
	"""
		Multiply the Hessian of the loss function wrt training variables by `v`.
	"""
	def hessian_vector_product(self, v):
		# Gradient-vector product
		elemwise_products = [
			tf.multiply(grad_elem, tf.stop_gradient(v_elem))
			for grad_elem, v_elem in zip(self.gradients, v)
			if grad_elem is not None
		]
		
		# Gradient of the GVP
		return tf.gradients(elemwise_products, self.var_list)
