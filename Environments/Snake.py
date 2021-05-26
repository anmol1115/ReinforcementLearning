import pygame
import random
import numpy as np
from pygame.math import Vector2

block_size = 20
num_block = 20
red = pygame.Color('red')
blue = pygame.Color('blue')
yellow = pygame.Color('yellow')

class Food:
	def __init__(self):
		self.randomize()

	def randomize(self, screen=None):
		x = random.randint(1, num_block-1)
		y = random.randint(1, num_block-1)
		self.pos = Vector2(x, y)
		if screen:
			self.draw(screen)

	def draw(self, screen):
		food_rect = pygame.Rect((int(self.pos.x)*block_size, int(self.pos.y)*block_size), (block_size, block_size))
		pygame.draw.rect(screen, red, food_rect)

class Snake:
	def __init__(self):
		self.reset()
		self.direction_list = [Vector2(1,0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
		
	def reset(self):
		self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
		self.direction = Vector2(1, 0)

	def draw_head(self, screen):
		head = self.body[0]
		head_rect = pygame.Rect((int(head.x*block_size), int(head.y*block_size)), (block_size, block_size))
		pygame.draw.rect(screen, blue, head_rect)

	def draw_body(self, screen):
		for body in self.body[1:]:
			body_rect = pygame.Rect((int(body.x*block_size), int(body.y*block_size)), (block_size, block_size))
			pygame.draw.rect(screen, yellow, body_rect)

	def draw(self, screen):
		self.draw_head(screen)
		self.draw_body(screen)

	def move(self, screen=None):
		body_copy = self.body[:-1]
		body_copy.insert(0, body_copy[0]+self.direction)
		self.body = body_copy
		if screen:
			self.draw(screen)

	def grow(self, screen=None):
		body_copy = self.body[:]
		body_copy.insert(0, body_copy[0]+self.direction)
		self.body = body_copy
		if screen:
			self.draw(screen)

	def turn(self, action):
		direction_idx = self.direction_list.index(self.direction)
		if action == 1:
			self.direction = self.direction_list[(direction_idx+1)%len(self.direction_list)]
		elif action == 2:
			self.direction = self.direction_list[(direction_idx-1)%len(self.direction_list)]
		elif action == 0:
			self.direction = self.direction

	def food_colision(self, f_obj, screen=None):
		reward = 0
		if self.body[0] == f_obj.pos:
			reward = 1
			self.grow(screen)
			f_obj.randomize(screen)
		return reward

	def collision(self, counter, point=None):
		if point is None:
			point = self.body[0]
		if point.x < 0 or point.y < 0 or point.x > num_block or point.y > num_block:
			return True
		if counter > 100*len(self.body) or point in self.body[1:]:
			return True
		return False

	def update(self, action, counter, f_obj, screen=None):
		rew_c = 0
		self.move(screen)
		self.turn(action)
		rew_f = self.food_colision(f_obj, screen)
		game_over= self.collision(counter)
		if game_over:
			rew_c = -1
		return game_over, rew_f+rew_c

class Env:
	s = Snake()
	f = Food()
	action_size = 3
	observation_size = 11
	def __init__(self):
		self.counter = 0
		self.screen = None
		self.clock =  pygame.time.Clock()

	def get_state(self):
		snake_head = self.s.body[0]
		point_l = Vector2(snake_head.x-1, snake_head.y)
		point_r = Vector2(snake_head.x+1, snake_head.y)
		point_u = Vector2(snake_head.x, snake_head.y-1)
		point_d = Vector2(snake_head.x, snake_head.y+1)

		direction = self.s.direction
		dir_l = direction == Vector2(-1, 0)
		dir_r = direction == Vector2(1, 0)
		dir_u = direction == Vector2(0, -1)
		dir_d = direction == Vector2(0, 1)

		state = [
		#Danger Ahead
		(dir_l and self.s.collision(0, point_l)) or
		(dir_r and self.s.collision(0, point_r)) or
		(dir_u and self.s.collision(0, point_u)) or
		(dir_d and self.s.collision(0, point_d)),
		#Danger Left
		(dir_l and self.s.collision(0, point_d)) or
		(dir_r and self.s.collision(0, point_u)) or
		(dir_u and self.s.collision(0, point_l)) or
		(dir_d and self.s.collision(0, point_r)),
		#Danger Right
		(dir_l and self.s.collision(0, point_u)) or
		(dir_r and self.s.collision(0, point_d)) or
		(dir_u and self.s.collision(0, point_r)) or
		(dir_d and self.s.collision(0, point_l)),

		dir_l, dir_r, dir_u, dir_d,

		#food pos
		(self.f.pos.x < snake_head.x),  
		(self.f.pos.x > snake_head.x),  
		(self.f.pos.y < snake_head.y),  
		(self.f.pos.y > snake_head.y),  
		]
		return np.array(state, dtype=int)

	def render(self):
		self.screen = pygame.display.set_mode((block_size*num_block, block_size*num_block))
		self.clock.tick(20)

	def step(self, action):
		self.counter += 1
		done, rew = self.s.update(action, self.counter, self.f, self.screen)
		if self.screen:
			self.f.draw(self.screen)
			pygame.display.flip()
		return self.get_state(), rew, done

	def reset(self):
		self.counter = 0
		self.s.reset()
		self.f.randomize(self.screen)
		return self.get_state()

	def sample(self):
		return random.randint(0, 2)

if __name__ == '__main__':
	env = Env()
	state = env.reset()
	for _ in range(100):
		env.render()
		new_state, reward, done = env.step(env.sample())
		# print(env.get_state())
		print(reward)