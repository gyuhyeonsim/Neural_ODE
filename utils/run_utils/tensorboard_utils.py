def draw_learning_curve(writer, id, loss, iter):
    writer.add_scalar('./loss/'+id, loss, iter)

