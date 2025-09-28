create extension if not exists vector;

create table if not exists documents (
  id bigserial primary key,
  doc_id text not null,
  chunk_id int not null,
  content text not null,
  metadata text,
  embedding vector(384)
);

create index if not exists documents_doc_idx on documents (doc_id);
create index if not exists documents_embed_idx on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);
