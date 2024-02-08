def generate():
    width = 556
    height =556
    randID:int =math.floor(random.random()*900)
    my_list:list[int] = []
    b1 = math.floor(random.random()*255)
    b2 = math.floor(random.random()*255)    
    for i in range(width):
        for j in range(height):
            if  j < b1 and i > b1:
                my_list.append((math.floor(random.random()*255),math.floor(random.random()*10),math.floor(random.random()*20)))
            elif j < b2 and i < b2:
                my_list.append((math.floor(random.random()*20),math.floor(random.random()*10),math.floor(random.random()*255)))
            elif i< b1:
                my_list.append((math.floor(random.random()*15),math.floor(random.random()*255),math.floor(random.random()*20)))
            else :
                 my_list.append((math.floor(random.random()*255),math.floor(random.random()*255),math.floor(random.random()*20)))

    img = Image.new('RGB', (width, height))
    img.putdata(my_list)
    img.save('/home/maeri_n/Flask_images/static/Images/ID.png')