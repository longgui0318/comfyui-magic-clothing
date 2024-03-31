def handle_block_info(block_key):
    block_level = block_key.split(".")
    if block_level[0] == "down_blocks":
        block_name = "input"
        block_number = int(block_level[1])
    elif block_level[0] == "mid_block":
        block_name = "middle"
        block_number = 0
    elif block_level[0] == "up_blocks":
        block_name = "output"
        block_number = int(block_level[1])
    else:
        block_name =None
        block_number = 0
    attention_index = 0
    for i,v in  enumerate(block_level):
        if  v == "attentions":
            attention_index = int(block_level[i+1])
            break
    block_number+=1
    return (block_name,block_number,attention_index)

def save_attn(value,attn_store,block_name,block_number,attention_index):
    if attn_store is None:
        return
    if block_name not in attn_store:
        attn_store[block_name]={}
    if block_number not in attn_store[block_name]:
        attn_store[block_name][block_number]={}
    attn_store[block_name][block_number][attention_index]=value