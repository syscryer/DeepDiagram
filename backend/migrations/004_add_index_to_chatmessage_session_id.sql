DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chatmessage_session_id') THEN
        CREATE INDEX idx_chatmessage_session_id ON chatmessage (session_id);
    END IF;
END $$;
