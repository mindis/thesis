import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import random
import operator
import time
import matplotlib.pyplot as plt
sns.set(style='ticks',color_codes=True)


def items_explored(path1,path2):

    items = pd.read_csv(path1)
    items1 = pd.read_csv(path2)
    items = pd.concat([items1,items])

    #make UNIX timestamp readable datetime
    times=[]
    for i in items['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i//1000.0))

    items['timestamp']=times
    return items

def events_explored(path):

    events = pd.read_csv(path)
    visitors = events["visitorid"].unique()
    times2 =[]
    for i in events['timestamp']:
        times2.append(datetime.datetime.fromtimestamp(i//1000.0))

    events['timestamp']=times2
    return events,visitors

def actions_plot(events):
    events_count = events["event"].value_counts()
    fig, axs = plt.subplots(ncols=2, figsize=(15, 8))
    sns.barplot(events_count.index, events_count.values, ax=axs[0])

    events_count = events["event"].value_counts()[1:]
    # plt.title('Actions Vs Count')
    g = sns.barplot(events_count.index, events_count.values, ax=axs[1])
    # g.set_yscale('log')
    events_count = events["event"].value_counts()[1:]
    plt.title('Add-to-cart V/s Transaction')
    sns.barplot(events_count.index, events_count.values)
    plt.show()

    print(events_count)


def create_dataframe(visitor_list,events):
    array_for_df = []
    for index in visitor_list:

        # Create that visitor's dataframe once
        v_df = events[events.visitorid == index]

        temp = []
        # Add the visitor id
        temp.append(index)

        # Add the total number of unique products viewed
        temp.append(v_df[v_df.event == 'view'].itemid.unique().size)

        # Add the total number of views regardless of product type
        temp.append(v_df[v_df.event == 'view'].event.count())

        # Add the total number of purchases
        number_of_items_bought = v_df[v_df.event == 'transaction'].event.count()
        temp.append(number_of_items_bought)

        # Then put either a zero or one if they made a purchase
        if (number_of_items_bought == 0):
            temp.append(0)
        else:
            temp.append(1)

        array_for_df.append(temp)

    return pd.DataFrame(array_for_df,
                        columns=['visitorid', 'num_items_viewed', 'view_count', 'bought_count', 'purchased'])

if __name__=='__main__':
    ITEMS1_CSV_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/item_properties_part1.csv'
    ITEMS2_CSV_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/item_properties_part2.csv'
    EVENTS_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/events.csv'
    events,visitors = events_explored(EVENTS_PATH)
    #items = items_explored(ITEMS1_CSV_PATH,ITEMS2_CSV_PATH)
    print(events.head())
    print(events.columns)
    # #print(items.head())
    # all_visitors = events.visitorid.sort_values().unique()
    # print(all_visitors.size)
    # buying_visitors = events[events.event == 'transaction'].visitorid.sort_values().unique()
    # print(buying_visitors.size)
    # buys_ratio = buying_visitors.size/all_visitors.size
    # print('buys/views percentage:{}%'.format(buys_ratio*100))
    # buying_visitors_df = create_dataframe(buying_visitors,events)
    # viewing_visitors_list = list(set(all_visitors) - set(buying_visitors))
    # random.shuffle(viewing_visitors_list)
    # viewing_visitors_df = create_dataframe(viewing_visitors_list[0:27820],events)
    # main_df = pd.concat([buying_visitors_df, viewing_visitors_df], ignore_index=True)
    # main_df = main_df.sample(frac=1)
    # print(main_df.head())
    # sns.pairplot(main_df, x_vars=['num_items_viewed', 'view_count', 'bought_count'],
    #              y_vars=['num_items_viewed', 'view_count', 'bought_count'], hue='purchased')

    items = events.itemid.value_counts()

    # plt.figure(figsize=(16, 9))
    # plt.hist(items.values, bins=10, log=True, color='red')
    # plt.xlabel('Number of times item appeared', fontsize=16)
    # plt.ylabel('Count of displays with item', fontsize=16)
    # #plt.show()
    corr = events[events.columns].corr()
    sns.heatmap(corr, annot=True)
    plt.show()
