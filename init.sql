-- Simplified Database Schema (No UUID extension required)
-- Uses SERIAL instead of UUID for primary keys

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    user_role VARCHAR(50) DEFAULT 'student',
    course_code VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id)
);

CREATE TABLE IF NOT EXISTS chat_logs (
    id SERIAL PRIMARY KEY,
    chat_id VARCHAR(255) NOT NULL,
    sandbox_session_id VARCHAR(255),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    user_email VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    mode VARCHAR(20) DEFAULT 'chat',
    title VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sandbox_environments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_config JSONB DEFAULT '{"temperature": 0.7}',
    created_by INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS sandbox_sessions (
    id SERIAL PRIMARY KEY,
    environment_id INTEGER REFERENCES sandbox_environments(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chat_logs_user_id ON chat_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_chat_id ON chat_logs(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON chat_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Enable Row Level Security
ALTER TABLE chat_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE sandbox_sessions ENABLE ROW LEVEL SECURITY;

-- Create Row Level Security policies
-- Note: These policies will work with integer IDs instead of UUIDs
CREATE POLICY IF NOT EXISTS chat_logs_user_policy ON chat_logs
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::INTEGER);

CREATE POLICY IF NOT EXISTS sandbox_sessions_user_policy ON sandbox_sessions
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::INTEGER);
