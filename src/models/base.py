class BaseVLMAdapter:
    def __init__(self, model_name, device="auto", cache_dir=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

    # def create_template(self, item):
    #     conversation = {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": item["image"]},
    #                 {"type": "text", "text": item["question"]},
    #             ],
    #         }
    #     return conversation

    def create_template(self, item):
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        raise NotImplementedError

    def infer(self, model, processor, inputs, max_new_tokens):
        raise NotImplementedError