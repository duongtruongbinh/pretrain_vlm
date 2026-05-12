import json
import unittest
from pathlib import Path


class VietnamTourismConversationTest(unittest.TestCase):
    def test_parse_new_conversation_schema(self):
        generator = __import__("scripts.generate_qa_vietnamtourism", fromlist=["parse_qa_response"])
        payload = {
            "description": "Một hội nghị trong hội trường.",
            "conversation": [
                {"role": "user", "content": "Ảnh này có gì nổi bật?"},
                {"role": "assistant", "content": "Ảnh cho thấy một hội nghị trong hội trường lớn."},
                {"role": "user", "content": "Chi tiết hội trường trông ra sao?"},
                {"role": "assistant", "content": "Hội trường có sân khấu, màn hình lớn và nhiều hàng ghế."},
            ],
        }

        description, conversation = generator.parse_qa_response(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(description, payload["description"])
        self.assertEqual(conversation, payload["conversation"])

    def test_parse_legacy_qa_pairs_schema(self):
        generator = __import__("scripts.generate_qa_vietnamtourism", fromlist=["parse_qa_response"])
        payload = {
            "description": "Một đoàn du khách trên tàu.",
            "qa_pairs": [
                {
                    "type": "description",
                    "question": "Trong ảnh có hoạt động gì?",
                    "answer": "Ảnh cho thấy du khách đang tham dự một hoạt động trên tàu.",
                }
            ],
        }

        _, conversation = generator.parse_qa_response(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(
            conversation,
            [
                {"role": "user", "content": "Trong ảnh có hoạt động gì?"},
                {
                    "role": "assistant",
                    "content": "Ảnh cho thấy du khách đang tham dự một hoạt động trên tàu.",
                },
            ],
        )

    def test_prepare_entry_keeps_article_context_out_of_training_messages(self):
        prepare = __import__("scripts.prepare_vietnamtourism_data", fromlist=["build_instruction_entry"])
        conversation = [
            {"role": "user", "content": "Ảnh này mô tả điều gì?"},
            {"role": "assistant", "content": "Ảnh mô tả một sự kiện du lịch."},
            {"role": "user", "content": "Sự kiện đó có ý nghĩa gì?"},
            {"role": "assistant", "content": "Nó góp phần quảng bá điểm đến địa phương."},
        ]

        entry = prepare.build_instruction_entry(
            image_path=Path("/tmp/sample.jpg"),
            conversation=conversation,
            system_prompt="System",
            image_id="abc",
            source="vietnamtourism",
            title="Quảng bá du lịch Việt Nam",
            caption="Du khách tham gia sự kiện.",
        )

        self.assertEqual(entry["messages"][1]["content"], "Ảnh này mô tả điều gì?")
        self.assertEqual(entry["messages"][3]["content"], "Sự kiện đó có ý nghĩa gì?")
        self.assertEqual(entry["title"], "Quảng bá du lịch Việt Nam")
        self.assertEqual(entry["caption"], "Du khách tham gia sự kiện.")
        self.assertEqual(entry["sample_type"], "conversation")


if __name__ == "__main__":
    unittest.main()
