python freeze_graph.py --input_graph=../model_pb/the_scream.pbtxt --input_checkpoint=../../models/generators/trained_model-20000 --output_graph=../freezed_models/the_scream.pb --output_node_names='generator/Slice'