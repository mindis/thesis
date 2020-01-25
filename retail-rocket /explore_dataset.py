import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import random
import operator
import time
import matplotlib.pyplot as plt
sns.set(style='ticks',color_codes=True)


def merge_items_csv(path1,path2):

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
    plt.title('Actions Vs Count')
    g = sns.barplot(events_count.index, events_count.values, ax=axs[1])
    g.set_yscale('log')
    events_count = events["event"].value_counts()[1:]
    plt.title('Add-to-cart Vs Transaction')

    sns.barplot(events_count.index, events_count.values)

    plt.show()

    print(events_count)

def top5itemsviewed_plot(grouped):
    views = grouped['view']
    count_view = {}
    # for item in set(views[:]):
    # print(item)
    #    count_view[item]=views.count(item)
    views = np.array(views[:])

    unique, counts = np.unique(views, return_counts=True)
    count_view = dict(zip(unique, counts))
    sorted_count_view = sorted(count_view.items(), key=operator.itemgetter(1), reverse=True)
    x = [i[0] for i in sorted_count_view[:5]]
    y = [i[1] for i in sorted_count_view[:5]]
    sns.barplot(x, y, order=x)
    plt.xlabel('itemid')
    plt.ylabel('number of appearences')
    plt.show()

def top5addtocart_plot(grouped):
    # the most addtocart itemid
    addtocart = grouped['addtocart']
    count_addtocart = {}
    # for item in set(addtocart[:]):
    #     #print(item)
    #     count_addtocart[item]=addtocart.count(item)
    addtocart = np.array(addtocart[:])
    unique, counts = np.unique(addtocart, return_counts=True)
    count_addtocart = dict(zip(unique, counts))

    sorted_count_addtocart = sorted(count_addtocart.items(), key=operator.itemgetter(1), reverse=True)
    x = [i[0] for i in sorted_count_addtocart[:5]]
    y = [i[1] for i in sorted_count_addtocart[:5]]
    sns.barplot(x, y, order=x)
    plt.xlabel('addtocart actions')
    plt.ylabel('number of appearences')
    plt.show()


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

# Write a function that would show items that were bought together (same of different dates) by the same customer
def recommender_bought_bought(item_id, purchased_items):

    # Perhaps implement a binary search for that item id in the list of arrays
    # Then put the arrays containing that item id in a new list
    # Then merge all items in that list and get rid of duplicates
    recommender_list = []
    for x in purchased_items:
        if item_id in x:
            recommender_list += x

        # Then merge recommender list and remove the item id
    recommender_list = list(set(recommender_list) - set([item_id]))

    return recommender_list


if __name__=='__main__':
    ITEMS1_CSV_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/item_properties_part1.csv'
    ITEMS2_CSV_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/item_properties_part2.csv'
    EVENTS_PATH = '/home/nick/Desktop/thesis/datasets/retail-rocket/events.csv'
    events,visitors = events_explored(EVENTS_PATH)
    #df = pd.read_csv('/home/nick/Desktop/thesis/datasets/retail-rocket/item_properties_part1.csv')
    #print(df)
    #items = items_explored(ITEMS1_CSV_PATH,ITEMS2_CSV_PATH)
    #print(items.head(20))
    print(events.head())
    print(events.columns)
    print(events.shape)

    # Get all unique visitors of the site
    all_visitors = events.visitorid.sort_values().unique()
    print('Number of visitors:{}'.format(all_visitors.size))
    #Get all visitors that completed a transaction
    buying_visitors = events[events.event == 'transaction'].visitorid.sort_values().unique()
    print('Number of buying visitors:{}'.format(buying_visitors.size))

    buys_ratio = buying_visitors.size/all_visitors.size
    print('buys/views percentage:{}%'.format(buys_ratio*100))

    #Get top 20 viewed/clicked/purchased items
    top20_items = events['itemid'].value_counts().head(20)
    print(top20_items)

    #Print only events made by user: 187946
    print(events.loc[events['itemid'] == 187946])

    actions_plot(events)
    #group events
    grouped = events.groupby('event')['itemid'].apply(list)
    top5itemsviewed_plot(grouped)
    top5addtocart_plot(grouped)

    customer_purchased = events[events.transactionid.notnull()].visitorid.unique()
    purchased_items = []
    # Create another list that contains all their purchases
    for customer in customer_purchased:
        # Generate a Pandas series type object containing all the visitor's purchases and put them in the list
        purchased_items.append(list(events.loc[(events.visitorid == customer) & (events.transactionid.notnull())].itemid.values))

    #print(purchased_items)
    #try recommender_bought_bought function with example itemid
    rec_list = recommender_bought_bought(302422, purchased_items)
    print(rec_list)









