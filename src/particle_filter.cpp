/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set the number of particles
	num_particles = 50;
	particles.reserve(num_particles);

	// Use the default random engine
	default_random_engine gen;

	// Create a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int n = 0; n < num_particles; n++)
	{
		Particle p;

		p.id = n;

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);

		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Use the default random engine
	default_random_engine gen;

	// Create a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto &p : particles)
	{
		// Bicycle motion model
		if (fabs(yaw_rate) > 0.0001)
		{
			p.x += (velocity / yaw_rate) * (sin((p.theta + yaw_rate * delta_t)) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos((p.theta + yaw_rate * delta_t)));
			p.theta = (p.theta + yaw_rate * delta_t);
		}
		else
		{
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto &obs : observations)
	{
		double min_dist = std::numeric_limits<double>::max();
		int min_id = -1;

		for (auto const &pred : predicted)
		{
			const auto d = dist(obs.x, obs.y, pred.x, pred.y);

			if (d < min_dist)
			{
				min_dist = d;
				min_id = pred.id;
			}
		}

		obs.id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (auto &p : particles)
	{
		// Find all landmarks within sensor range for this particle
		std::vector<LandmarkObs> in_range_landmarks;
		for (auto landmark : map_landmarks.landmark_list)
		{
			if (fabs(landmark.x_f - p.x) <= sensor_range && fabs(landmark.y_f - p.y) <= sensor_range)
			{
				LandmarkObs lm;
				lm.id = landmark.id_i;
				lm.x = landmark.x_f;
				lm.y = landmark.y_f;
				in_range_landmarks.push_back(lm);
			}
		}

		// Transform observations to map coordinates
		std::vector<LandmarkObs> map_observations;
		map_observations.reserve(observations.size());

		for (const auto &obs : observations)
		{
			double map_x = p.x + (obs.x * cos(p.theta)) - (obs.y * sin(p.theta));
			double map_y = p.y + (obs.x * sin(p.theta)) + (obs.y * cos(p.theta));

			map_observations.push_back(LandmarkObs{obs.id, map_x, map_y});
		}

		// Do data association
		dataAssociation(in_range_landmarks, map_observations);

		p.weight = 1.0;

		const auto std_x = std_landmark[0];
		const auto std_y = std_landmark[1];

		const auto norm_factor = (1.0 / (2 * M_PI * std_x * std_y));
		double exp_x, exp_y;

		for (auto obs : map_observations)
		{
			const auto x = obs.x;
			const auto y = obs.y;

			double m_x, m_y;
			for (auto l : in_range_landmarks)
			{
				if (l.id == obs.id)
				{
					m_x = l.x;
					m_y = l.y;

					exp_x = (x - m_x) * (x - m_x) / (2.0 * std_x * std_x);
					exp_y = (y - m_y) * (y - m_y) / (2.0 * std_y * std_y);
					p.weight *= norm_factor * exp(-(exp_x + exp_y));
				}
			}
		}
	}
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Use the default random engine
	default_random_engine gen;

	// Extract all weights.
	std::vector<double> weights;
	weights.reserve(num_particles);
	for (const auto &p : particles)
	{
		weights.push_back(p.weight);
	}

	std::discrete_distribution<int> dist_w(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	new_particles.reserve(num_particles);
	for (auto i = 0; i < num_particles; i++)
	{
		new_particles.push_back(particles[dist_w(gen)]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations, const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}