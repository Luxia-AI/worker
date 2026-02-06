// ====== RESET (DATA ONLY) ======
MATCH (n)
DETACH DELETE n;

// ====== CONSTRAINTS ======
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT relation_rid_unique IF NOT EXISTS
FOR (r:Relation)
REQUIRE r.rid IS UNIQUE;

CREATE CONSTRAINT source_url_unique IF NOT EXISTS
FOR (s:Source)
REQUIRE s.url IS UNIQUE;

// Optional Claim constraint
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim)
REQUIRE c.id IS UNIQUE;

// ====== INDEXES ======
CREATE INDEX entity_name_index IF NOT EXISTS
FOR (e:Entity)
ON (e.name);

CREATE INDEX relation_predicate_index IF NOT EXISTS
FOR (r:Relation)
ON (r.predicate);

CREATE INDEX source_domain_index IF NOT EXISTS
FOR (s:Source)
ON (s.domain);

CREATE INDEX relation_confidence_index IF NOT EXISTS
FOR (r:Relation)
ON (r.confidence);

CREATE INDEX relation_updated_at_index IF NOT EXISTS
FOR (r:Relation)
ON (r.updated_at);

// Optional Claim index
CREATE INDEX claim_text_index IF NOT EXISTS
FOR (c:Claim)
ON (c.text);

// ====== DONE ======
RETURN "KG reset + schema created" AS status;