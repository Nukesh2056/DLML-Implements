from utils import create_data, calculate_max_len, pad_data


a =create_data("Hi My name is Nukes, My Inspiration is no one, I'm pretty much going to be hopeless in future, with my own hope & ambitions, sometimes I behave like a drunk guy, even though I don't drink, this is because sometimes, I just want to be dead on my bed, millions KM's away from earth where, there are no idiots who can torture my mental health")

# for i in a:
#     print(i)
    

max_len = calculate_max_len(a)
# print(max_len)

padded_data = pad_data(a,max_len )
print(padded_data)