import json
import tempfile
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

    def test_save_batch_result_text_appends_only_new_records(self):
        generator = __import__("scripts.generate_qa_vietnamtourism", fromlist=["save_batch_result_text"])
        response_payload = {
            "description": "Một nhóm du khách đang đứng trước cổng.",
            "conversation": [
                {"role": "user", "content": "Trong ảnh có mấy người nổi bật?"},
                {"role": "assistant", "content": "Có ba người nổi bật ở phía trước ảnh."},
            ],
        }
        result_text = "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "img-existing",
                        "response": {
                            "body": {
                                "choices": [
                                    {"message": {"content": json.dumps(response_payload, ensure_ascii=False)}}
                                ]
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "custom_id": "img-new",
                        "response": {
                            "body": {
                                "choices": [
                                    {"message": {"content": json.dumps(response_payload, ensure_ascii=False)}}
                                ]
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        records = {
            "img-existing": {
                "image_id": "existing",
                "image_path": "/tmp/existing.jpg",
                "title": "Tựa đề cũ",
                "caption": "Caption cũ",
                "article_url": "https://example.com/old",
                "date": "2026-05-12",
            },
            "img-new": {
                "image_id": "new",
                "image_path": "/tmp/new.jpg",
                "title": "Tựa đề mới",
                "caption": "Caption mới",
                "article_url": "https://example.com/new",
                "date": "2026-05-12",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "batch_results.jsonl"
            done_ids = {"existing"}

            saved = generator.save_batch_result_text(result_text, records, output_path, done_ids)

            self.assertEqual(saved, 1)
            self.assertEqual(done_ids, {"existing", "new"})
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            saved_record = json.loads(lines[0])
            self.assertEqual(saved_record["image_id"], "new")
            self.assertEqual(saved_record["description"], response_payload["description"])


if __name__ == "__main__":
    unittest.main()
