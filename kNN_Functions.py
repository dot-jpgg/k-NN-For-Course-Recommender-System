# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:34:05 2021

@author: liamg

DATA417 kNN Functions
"""

# Importing packages.

import numpy as np
import numpy.random as npr
import pandas as pd
from os import listdir
from dill import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global constants.

CHOSEN_ONE = 1 # The Chosen One's ID.
PATH = r"C:\Users\liamg\OneDrive\Desktop\University\2021\DATA417\Group Project\Data" + "\\"

#%%

# Liam's Functions.


# Writing functions.

def get_chosen_ones_courses(people_dict):
    """
    Tessa.
    """
    # Getting thee Chosen One's metric DataFrame (index [1]).
    chosen_one_metric_df = people_dict[CHOSEN_ONE][1]
    
    # Getting their courses.
    return set(chosen_one_metric_df["Course"])
    

def compute_weights(courses_removed, total_courses):
    """
    Given the courses removed, this function returns the weights to apply to
    the similarities.
    """
    return (total_courses - courses_removed) / total_courses


def get_metric_vector(person, chosen_ones_courses):
    """
    Given a person's course matrix and thee Chosen One's courses, this
    function returns a metric dictionary of the form,
    
    metric_dict = {course1 : m_vec1, course2 : m_vec2, etc...}
    
    where metric vectors are of the form,
        
    metric_vector = [curr_course["Rev-A enjoyment"],
                     curr_course["Rev-B easiness"]]
    
    Additionally, this function returns weights which are used in calculating
    the similarities.
    """
    # Creating counters for the person's weights.
    courses_removed = 0
    total_courses = 0
    
    # Creating a dictionary for the person's metrics.
    metric_dict = dict()
    
    # Iterating through the other person's courses.
    for i, row in person.iterrows():
        # Increasing number of total courses.
        total_courses += 1
        
        # Current course we're considering.
        curr_course = row["Course"]
        
        # Has thee Chosen One has taken this course?
        if curr_course in chosen_ones_courses:
            
            # Creating metric vector.
            curr_metrics = [row["Rev-A enjoyment"],
                            row["Rev-B easiness"],
                            row["Grade"]]
            
            # Adding metric vector to metric dictionary.
            metric_dict[curr_course] = curr_metrics
        else:
            courses_removed += 1
    
    weights = compute_weights(courses_removed, total_courses)
    
    return metric_dict, weights
    
    
def compute_similarity(chosen_one_metric_dict, chosen_ones_courses, other_metric_dict, other_weights):
    """
    Given thee Chosen One's metric dictionary & weights and another person's
    metric dictionary & weights, this function returns the similarity between
    the two people.
    """
    ##### CASE (1): There are no courses in common. #####
    
    if other_weights == 0:
        return 0
    
    ##### CASE (2): There are courses in common, but they aren't identical. #####
    
    # Similarity to add to.
    similarity = 0
    
    # Iterating through the courses.
    for course in other_metric_dict.keys():
        
        # Does thee Chosen One also take this course?
        if course in chosen_ones_courses:
            # Computing current similarity.
            c_vector = chosen_one_metric_dict[course]
            o_vector = other_metric_dict[course]
            curr_diff = np.array(c_vector) - np.array(o_vector)
            curr_dist = np.sqrt(curr_diff.dot(curr_diff)) # Taking magnitude.
            
            ##### CASE (3): Identical rating. #####
            
            if curr_dist == 0:
                similarity += 1
            else:
                similarity += 1 / curr_dist
            
    # Scaling with weights.
    similarity = similarity * other_weights
    
    # Normalizing by number of thee Chosen One's courses.
    similarity = similarity / len(chosen_ones_courses)
    
    return similarity


def get_similarities(people_dict, chosen_ones_courses):
    """
    Given a dictionary of people's DataFrames and a set containing thee Chosen
    One's courses, this function coordinates a series of other functions to
    return a vector containing the similarities between all others in our
    data set, and thee Chosen One.
    """
    # Similarity dictionary.
    similarity_dict = dict()
    
    # Getting thee Chosen One.
    chosen_ones_demo, chosen_one = people_dict[CHOSEN_ONE]
    
    # Getting thee Chosen One's course metrics.
    chosen_one_metric_dict, chosen_ones_weights = \
        get_metric_vector(chosen_one, chosen_ones_courses)
        
    # Iterating through people in our data set.
    for id_num in people_dict.keys():
        
        # Is this thee Chosen One?
        if not(id_num == CHOSEN_ONE):
            
            # Retrieving the other person's data.
            o_demo, other_person = people_dict[id_num]
            
            # Getting other person's metric dictionary.
            other_metric_dict, other_weights = \
                get_metric_vector(other_person, chosen_ones_courses)
            
            # Computing the similarity.
            similarity = compute_similarity(chosen_one_metric_dict,
                                            chosen_ones_courses,
                                            other_metric_dict,
                                            other_weights)
            
            # Adding to similarity dictionary.
            similarity_dict[id_num] = similarity
    
    return similarity_dict
    
    
def produce_test_data():
    """
    This function produces some basic test data.
    """
    # Producing simulated column data.
    test1 = {"Course" : ["COSC121", "MATH101"], "Rev-A enjoyment" : [1, 2], "Rev-B easiness" : [3, 4]}
    test2 = {"Course" : ["COSC121", "MATH102"], "Rev-A enjoyment" : [2, 2], "Rev-B easiness" : [4, 4]}
    test3 = {"Course" : ["COSC121", "MATH102"], "Rev-A enjoyment" : [1, 2], "Rev-B easiness" : [3, 4]}
    
    # Turning into a dictionary.
    people_dict = {123 : [None, pd.DataFrame.from_dict(test1)],
                   456 : [None, pd.DataFrame.from_dict(test2)],
                   789 : [None, pd.DataFrame.from_dict(test3)]}
    
    # Returning result.
    return people_dict


def convert_similarity_dict_to_df(similarity_dict):
    """
    This function converts the similarity dictionary to a DataFrame.
    """
    # Empty dictionary to append to.
    new_similarity_dict = {"Student_id" : [CHOSEN_ONE], "Similarity" : [np.inf]}
    
    # Iterating through keys (IDs) of dictionary.
    for id_num in similarity_dict.keys():
        # Adding ID num and similarity to the columns in the dict.
        new_similarity_dict["Student_id"].append(id_num)
        new_similarity_dict["Similarity"].append(similarity_dict[id_num])
    
    # Converting to DataFrame.
    similarity_df = pd.DataFrame.from_dict(new_similarity_dict)
    
    # Returning DataFrame
    return similarity_df   


def join_similarity_to_metric(similarity_dict):
    """
    This function joins the similarity dictionary to the metric dictionary and
    returns the result.
    """
    # Get metric DataFrame (Tessa's function).
    metric_df = get_metric_df()
    
    # Converting similarity dict. to DataFrame.
    similarity_df = convert_similarity_dict_to_df(similarity_dict)
    
    # Joining metric DataFrame to similarity DataFrame.
    metric_df = metric_df.merge(similarity_df)
    
    # Returning result.
    return metric_df


def save_metric_df_as_csv(metric_df, filename, path=""):
    """
    Given the new metric dictionary (with similarities), this function saves
    it as a CSV with the given filename.
    """
    metric_df.to_csv(path + filename, index=False)
    
    
def get_metric_dict(people_dict, chosen_ones_courses):
    """
    Given a people dictionary, this function does a similar job to the
    "get_similarities" function, only, instead of computing the similarities,
    it stores the metric information in a metric dictionary.
    """
    # Metric dictionary.
    metric_dict = dict()
    
    # Getting thee Chosen One.
    chosen_ones_demo, chosen_one = people_dict[CHOSEN_ONE]
    
    # Getting thee Chosen One's course metrics.
    chosen_one_metric_dict, chosen_ones_weights = \
        get_metric_vector(chosen_one, chosen_ones_courses)
        
    # Iterating through people in our data set.
    for id_num in people_dict.keys():
        
        # Is this thee Chosen One?
        if not(id_num == CHOSEN_ONE):
            
            # Retrieving the other person's data.
            o_demo, other_person = people_dict[id_num]
            
            # Getting other person's metric dictionary.
            other_metric_dict, other_weights = \
                get_metric_vector(other_person, chosen_ones_courses)
                
            # If this person has courses in common:
            if other_metric_dict != {}:
                # Adding to dictionary.
                metric_dict[id_num] = other_metric_dict
    
    # Returning metric dict.
    return metric_dict
    
    
def draw_points(people_dict, chosen_ones_courses):
    """
    Given the people_dict, the chosen_ones_courses, the course chosen and the
    person who 'supplied this recommendation', this functions gets a
    metric dictionary which contains everyone's points in the form:
        {ID : {Course Code : [Metric_1, Metric_2, Metric_3],
               Course Code : etc...}
        }
    Then it draws the point for everyone with this course in 3D space,
    highlighting thee Chosen One's course
    """
    # Getting metric dictionary.
    metric_dict = get_metric_dict(people_dict, chosen_ones_courses)
    
    # Getting thee Chosen One's metric dictionary.
    chosen_ones_demo, chosen_one = people_dict[CHOSEN_ONE]
    chosen_one_metric_dict, chosen_ones_weights = \
        get_metric_vector(chosen_one, chosen_ones_courses)
    print(metric_dict)
    print(chosen_one_metric_dict)
    s
    
    # Setting up axes.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True
    fig = plt.figure(1, dpi=400, figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    angle = 310
    ax.set_title("Course Space for {0}, {1}, and {2}".format(*chosen_ones_courses))
    ax.view_init(azim=angle)
    ax.set_xlabel("Enjoyment")
    ax.set_ylabel("Easiness")
    ax.set_zlabel("Grade")
    
    # Plotting thee Chosen One in red.
    
    
    # Iterating through values.
    for point in metric_dict.values():
        print(point)
        print(*point)
        ax.scatter(*point)
        
    
def jitter(point, scale=5):
    """
    Given a point (for our purposes, this point is assumed to be of the
    magnitude ||V||_inf = 5) this function adds jitter to that point.
    """
    return [ax + npr.random() / scale for ax in point]
    
    
    
    
    

# Main function.

def main():
    """
    Main function.
    """
    # Printing updates.
    print("Starting algorithm...")
    
    # # Gathering a people dictionary (Tessa's Functions)
    # print("... Getting people dictionary...")
    # people_dict = get_dictionaries()
    
    # # Saving people dictionary.
    # print("... Saving people dictionary...")
    # filename = r"people_dict.pkl"
    # with open(filename, 'wb') as handle:
    #     dump(people_dict, handle, protocol=HIGHEST_PROTOCOL)
    
    # Loading people dictionary.
    print("... Loading people dictionary...")
    filename = r"people_dict.pkl"
    with open(filename, 'rb') as handle:
        people_dict = load(handle)
    
    # Creating the Chosen One's courses.
    print("... Getting thee Chosen One's courses...")
    chosen_ones_courses = get_chosen_ones_courses(people_dict)
    
    # Get similarities.
    print("... Computing similarities...")
    similarity_dict = get_similarities(people_dict, chosen_ones_courses)

    # Join similarities to metric DataFrame.
    print("... Joining similarity data to metric data...")
    metric_df = join_similarity_to_metric(similarity_dict)
    
    # Saving as CSV.
    print("... Saving as CSV...")
    filename = "metric_df.csv"
    save_metric_df_as_csv(metric_df, filename)
    
    # Displaying image.
    print("... Drawing points in 3D space...")
    draw_points(people_dict, chosen_ones_courses)
    
    # Printing updates.
    print("... Done!")
    

#%%


# Tessa's Function's

def read_data(filename):
    '''a'''
    df = pd.read_csv(filename)
    return df

def construct_dictionaries (df1, df2):
    '''Makes slice dictionaries'''
    result_dictionary = dict()
    construct_slice_dfs(df1, df2)
    dfs_list, unique_list_id = construct_slice_dfs(df1, df2)
    i = 0
    relevant_data_df1 = make_dataframes(df1)
    for user_id in unique_list_id:
        result_dictionary[user_id] = [relevant_data_df1[i], dfs_list[i]]
        i += 1
    return result_dictionary

def construct_slice_dfs(df1, df2):
    '''Creates new data frame with all of a students info stored as a dictionary
    by student ID'''
    
    list_id = df1["Student_id"].tolist()   
    relevant_data_df2 = make_dataframes(df2)
    unique_list_id = create_unique_listid(list_id)  
    all_people_list = list()    
    dfs_list = list()
    for user_id in unique_list_id:
        #
        if user_id % 1000 == 0:
            print(user_id)
        #
        person_courses_info_list = list()
        person_courses_info_list.append(user_id)
        for data in relevant_data_df2:
            if data[0] == user_id:
                person_courses_info_list.append(data[1:])
        all_people_list.append(person_courses_info_list)
        number_courses_taken = len(person_courses_info_list) - 1
        column_strings = tuple(["Course", "Grade", "Rev-A enjoyment", "Rev-B easiness"])
        df_entries = person_courses_info_list[1:]
        personal_df = pd.DataFrame(columns = column_strings)
        i = 0
        while i < (len(df_entries)):
            personal_df.loc[i] = df_entries[i]
            i += 1
        dfs_list.append(personal_df)
    return dfs_list, unique_list_id
    
        
def create_unique_listid(list_id):
    '''creates list of only unique ids'''
    unique_list_id = list()
    for different_id in list_id:
        if different_id not in unique_list_id:
            unique_list_id.append(different_id)
    return unique_list_id
 
    
def make_dataframes(df):
    '''I dunno'''
    list_df = df.values.tolist()
    relevant_data_list = list()
    for entry in list_df:
        relevant_data = entry[0:]
        relevant_data_list.append(relevant_data)
    return relevant_data_list


def get_dictionaries():
    df1 = read_data(PATH + "a_demographics.csv")
    df2 = read_data(PATH + "b_metrics.csv")
    result_dictionary = construct_dictionaries(df1, df2)
    return result_dictionary


def get_metric_df():
    """
    This function retrieves the metric DataFrame.
    """
    return read_data(PATH + "b_metrics.csv")
    
    


#%%
    
# Running code.

if __name__ == '__main__':
    main()
    # result_dictionary = get_dictionaries()
    # path = r"C:\Users\liamg\OneDrive\Desktop\University\2021\DATA417\Group Project\Data" + "\\"
    # get_dictionary(path)